# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import json
import logging
import os
import re
import uuid
from io import StringIO
from typing import Any, ClassVar, Dict, Optional

from fastapi import FastAPI
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)

# Define allowed reasoning tags for axiom reward
ALLOWED_TAGS = {
    "def", "definition", "defention",
    "axiom", "axioms",
    "theorem",
    "property",
    "calculation",
    "fact", "facts"
}


class LibraryJudgeMathResourcesServerConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    should_use_judge: bool = True
    
    # Multi-component reward weights
    format_weight: float = 0.1
    outcome_weight: float = 2.0
    axiom_weight: float = 0.1
    coherence_weight: float = 0.2
    relevance_weight: float = 0.1
    efficiency_weight: float = 0.05
    
    # Difficulty scaling parameters
    use_difficulty_scaling: bool = True
    difficulty_scaling_cap: float = 4.0
    
    # Trace logging settings
    save_traces: bool = True
    trace_dir: str = "logs/training_traces"


class LibraryJudgeMathRunRequest(BaseRunRequest):
    question: str
    expected_answer: str
    llama8b_solve_rate: Optional[float] = 0.5
    domain: Optional[str] = "unknown"
    source: Optional[str] = "unknown"


class LibraryJudgeMathVerifyRequest(LibraryJudgeMathRunRequest, BaseVerifyRequest):
    pass


class JudgeEvaluation(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse
    outcome_correct: Optional[float] = None
    reasoning_coherence: Optional[float] = None
    reasoning_relevance: Optional[float] = None
    format_compliance: Optional[float] = None
    tag_usage: Optional[float] = None
    efficiency: Optional[float] = None


class LibraryJudgeMathVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_answer: Optional[str]
    library_reward: float
    judge_evaluations: Optional[list[JudgeEvaluation]]
    
    # Individual reward components for transparency
    format_reward: Optional[float] = None
    outcome_reward: Optional[float] = None
    axiom_reward: Optional[float] = None
    coherence_reward: Optional[float] = None
    relevance_reward: Optional[float] = None
    efficiency_reward: Optional[float] = None
    difficulty_multiplier: Optional[float] = None
    scaled_outcome_reward: Optional[float] = None


class LibraryJudgeMathResourcesServer(SimpleResourcesServer):
    # Enhanced judge prompt for multi-component scoring
    JUDGE_SYSTEM_MESSAGE: ClassVar[str] = """You are a mathematical reasoning validator. 
Evaluate the following student response to a problem.

CRITERIA:
1. outcome_correct (0.0 or 1.0): Is the final answer mathematically equivalent to the ground truth? (e.g., "1/2" == "0.5")
2. reasoning_coherence (0.0 to 1.0): Does the thinking process follow a logical path without contradictions?
3. reasoning_relevance (0.0 to 1.0): Does the reasoning directly address the problem asked?
4. format_compliance (0.0 or 1.0): Does the student use <think> tags for reasoning and \\boxed{{}} for the final answer?
5. tag_usage (0.0 to 1.0): Evaluate the use of mathematical reasoning tags.
   
   ALLOWED TAGS (case-insensitive, with brackets like [Tag]):
   - [Def] or [Definition] - For defining terms or concepts
   - [Axiom] or [Axioms] - For stating fundamental assumptions
   - [Theorem] - For referencing or applying theorems
   - [Property] - For mathematical properties being used
   - [Calculation] - For explicit computational steps
   - [Fact] or [Facts] - For stating known mathematical facts
   
   SCORING GUIDE:
   - 1.0: The tag correctly identifies a valid mathematical step, definition, axiom, calculation, theorem, or property that is directly used in the reasoning OR references something taken from the question.
   - 0.5: The tag is technically correct but the content doesn't actually use or apply what the tag claims. It's just stated but not meaningfully connected.
   - 0.0: The tag refers to a non-existent theorem/axiom OR the mathematical logic in the statement is false.

6. efficiency (0.0 to 1.0): Is the reasoning concise and focused, or is there unnecessary fluff?

Return ONLY a JSON object with these exact keys."""

    JUDGE_PROMPT_TEMPLATE: ClassVar[str] = """PROBLEM:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

STUDENT RESPONSE:
{response}

JSON RESPONSE FORMAT:
{{
  "outcome_correct": float,
  "reasoning_coherence": float,
  "reasoning_relevance": float,
  "format_compliance": float,
  "tag_usage": float,
  "efficiency": float
}}
Return ONLY this JSON object."""

    # Keep legacy labels for backward compatibility
    JUDGE_EQUAL_LABEL: ClassVar[str] = "[[A=B]]"
    JUDGE_NOT_EQUAL_LABEL: ClassVar[str] = "[[A!=B]]"

    config: LibraryJudgeMathResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self._library_verifier = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )
        
        # Initialize trace logging
        if self.config.save_traces:
            os.makedirs(self.config.trace_dir, exist_ok=True)
            self.trace_file = os.path.join(
                self.config.trace_dir,
                f"trace_{uuid.uuid4().hex[:8]}.jsonl"
            )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Additional server routes go here! e.g.:
        # app.post("/get_weather")(self.get_weather)

        return app

    async def verify(self, body: LibraryJudgeMathVerifyRequest) -> LibraryJudgeMathVerifyResponse:
        """
        Main verification method that orchestrates all reward components.
        """
        assistant_responses = []
        for output_item in body.response.output:
            if output_item.type != "message":
                continue

            for content_item in output_item.content:
                if content_item.type != "output_text":
                    continue

                assistant_responses.append(content_item.text)

        combined_response = "".join(assistant_responses)
        
        # Compute all reward components
        result = await self._verify_answer(
            body.question,
            body.expected_answer,
            combined_response,
            body.llama8b_solve_rate,
            body.domain,
            body.source
        )
        
        return LibraryJudgeMathVerifyResponse(
            **body.model_dump(),
            reward=result["reward"],
            extracted_answer=result["extracted_answer"],
            library_reward=result["library_reward"],
            judge_evaluations=result["judge_evaluations"],
            format_reward=result["format_reward"],
            outcome_reward=result["outcome_reward"],
            axiom_reward=result["axiom_reward"],
            coherence_reward=result.get("coherence_reward"),
            relevance_reward=result.get("relevance_reward"),
            efficiency_reward=result.get("efficiency_reward"),
            difficulty_multiplier=result["difficulty_multiplier"],
            scaled_outcome_reward=result["scaled_outcome_reward"],
        )

    async def _verify_answer(
        self,
        question: str,
        expected_answer: str,
        generated_answer: str,
        llama8b_solve_rate: float = 0.5,
        domain: str = "unknown",
        source: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Comprehensive verification with multi-component rewards and difficulty scaling.
        
        Returns a dictionary with all reward components and metadata.
        """
        # Initialize reward components
        format_reward = self._compute_format_reward(generated_answer)
        axiom_reward = self._compute_axiom_reward(generated_answer)
        difficulty_multiplier = self._compute_difficulty_multiplier(llama8b_solve_rate)
        
        # Get library-based verification
        library_reward, extracted_answer = self._verify_answer_with_library(
            expected_answer, generated_answer
        )
        
        # Initialize variables
        outcome_reward = 0.0
        scaled_outcome_reward = 0.0
        coherence_reward = None
        relevance_reward = None
        efficiency_reward = None
        judge_evaluations = None
        
        # Use LLM judge if enabled and ground truth is available
        if self.config.should_use_judge and expected_answer.strip():
            try:
                judge_evaluation = await self._verify_answer_with_judge(
                    question, expected_answer, generated_answer
                )
                judge_evaluations = [judge_evaluation]
                
                # Extract multi-component scores from judge
                outcome_correct = judge_evaluation.outcome_correct or 0.0
                outcome_reward = outcome_correct * self.config.outcome_weight
                scaled_outcome_reward = outcome_reward * difficulty_multiplier
                
                # Additional components from judge
                coherence_reward = (
                    (judge_evaluation.reasoning_coherence or 0.0) * self.config.coherence_weight
                )
                relevance_reward = (
                    (judge_evaluation.reasoning_relevance or 0.0) * self.config.relevance_weight
                )
                efficiency_reward = (
                    ((judge_evaluation.efficiency or 0.5) - 0.5) * self.config.efficiency_weight * 2
                )
                
            except Exception as e:
                # Fallback to library reward if judge fails
                outcome_reward = library_reward * self.config.outcome_weight
                scaled_outcome_reward = outcome_reward * difficulty_multiplier
        else:
            # Fallback: use library reward
            outcome_reward = library_reward * self.config.outcome_weight
            scaled_outcome_reward = outcome_reward * difficulty_multiplier
        
        # Calculate total reward
        total_reward = format_reward + scaled_outcome_reward + axiom_reward
        if coherence_reward is not None:
            total_reward += coherence_reward
        if relevance_reward is not None:
            total_reward += relevance_reward
        if efficiency_reward is not None:
            total_reward += efficiency_reward
        
        # Prepare reward components dictionary
        reward_components = {
            "format_reward": format_reward,
            "outcome_reward": outcome_reward,
            "scaled_outcome_reward": scaled_outcome_reward,
            "axiom_reward": axiom_reward,
            "difficulty_multiplier": difficulty_multiplier,
        }
        if coherence_reward is not None:
            reward_components["coherence_reward"] = coherence_reward
        if relevance_reward is not None:
            reward_components["relevance_reward"] = relevance_reward
        if efficiency_reward is not None:
            reward_components["efficiency_reward"] = efficiency_reward
        
        # Difficulty info for logging
        difficulty_info = {
            "llama8b_solve_rate": llama8b_solve_rate,
            "difficulty_multiplier": difficulty_multiplier,
            "domain": domain,
            "source": source
        }
        
        # Log the trace
        self._log_trace(
            question=question,
            response=generated_answer,
            ground_truth=expected_answer,
            reward=total_reward,
            reward_components=reward_components,
            difficulty_info=difficulty_info
        )
        
        return {
            "reward": total_reward,
            "extracted_answer": extracted_answer,
            "library_reward": library_reward,
            "judge_evaluations": judge_evaluations,
            "format_reward": format_reward,
            "outcome_reward": outcome_reward,
            "axiom_reward": axiom_reward,
            "coherence_reward": coherence_reward,
            "relevance_reward": relevance_reward,
            "efficiency_reward": efficiency_reward,
            "difficulty_multiplier": difficulty_multiplier,
            "scaled_outcome_reward": scaled_outcome_reward,
        }

    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield
    
    def _compute_format_reward(self, response: str) -> float:
        """Reward the use of <think> and \boxed{} tags."""
        has_think = "<think>" in response and "</think>" in response
        has_boxed = "\\boxed{" in response
        
        score = 0.0
        if has_think:
            score += 0.5
        if has_boxed:
            score += 0.5
        
        return score * self.config.format_weight
    
    def _compute_axiom_reward(self, response: str) -> float:
        """Count valid reasoning tags in the response."""
        tag_pattern = r'\[([^\]]+)\]'
        matches = re.findall(tag_pattern, response, re.IGNORECASE)
        valid_tags = [m for m in matches if m.lower() in ALLOWED_TAGS]
        
        # Scale: 0 tags = 0.0, 1-2 tags = 0.5, 3+ tags = 1.0 (max)
        if len(valid_tags) == 0:
            return 0.0
        elif len(valid_tags) <= 2:
            return self.config.axiom_weight * 0.5
        else:
            return self.config.axiom_weight
    
    def _compute_difficulty_multiplier(self, llama8b_solve_rate: float) -> float:
        """
        Compute reward multiplier based on problem difficulty.
        
        llama8b_solve_rate: 0.0 (hardest) to 1.0 (easiest)
        Returns: multiplier from 1.0 to difficulty_scaling_cap
        
        Formula: 1.0 / llama8b_solve_rate, capped at difficulty_scaling_cap
        """
        if not self.config.use_difficulty_scaling:
            return 1.0
        
        # Clamp solve_rate to avoid division by zero (minimum 0.05 = 20x max)
        solve_rate = max(llama8b_solve_rate, 0.05)
        
        # Calculate multiplier: easier problems get lower multiplier
        multiplier = 1.0 / solve_rate
        
        # Cap the multiplier to avoid extreme rewards
        return min(multiplier, self.config.difficulty_scaling_cap)
    
    def _log_trace(
        self,
        question: str,
        response: str,
        ground_truth: str,
        reward: float,
        reward_components: Dict[str, float],
        difficulty_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Append a single generation event to the trace file."""
        if not self.config.save_traces:
            return
        
        try:
            trace_entry = {
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "reward": float(reward),
                "reward_components": reward_components,
                "difficulty": difficulty_info,
                "timestamp": str(uuid.uuid4())
            }
            
            with open(self.trace_file, "a") as f:
                f.write(json.dumps(trace_entry) + "\n")
        except Exception as e:
            # Silently fail to avoid disrupting training
            pass

    def _verify_answer_with_library(self, expected_answer: str, generated_answer: str) -> tuple[float, Optional[str]]:
        # This functionality is migrated from Nemo RL.
        # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
        try:
            ground_truth_parsable = "\\boxed{" + expected_answer + "}"
            with self._mute_output():
                ret_score, extracted_answer = self._library_verifier([ground_truth_parsable], [generated_answer])

            reward = float(ret_score)

            if extracted_answer is not None:
                # Make sure the extracted answer has two elements.
                assert len(extracted_answer) == 2

                extracted_gold, extracted_prediction = extracted_answer

                # Get the extracted answer.
                for pred in extracted_prediction:
                    if any(grader.verify(gold, pred) for gold in extracted_gold):
                        extracted_answer = pred
                        break
                else:
                    # If no match is found, that means all the answers are
                    # incorrect.  The first prediction is used as the extracted
                    # answer.
                    extracted_answer = extracted_prediction[0]

            return reward, extracted_answer

        # It's possible to emit a TimeoutException and that wouldn't be caught since
        # it actually subclasses from BaseException and math-verify itself does not
        # catch it.
        except (Exception, TimeoutException):
            return 0.0, None

    async def _verify_answer_with_judge(
        self, question: str, expected_answer: str, generated_answer: str
    ) -> tuple[Dict[str, float], JudgeEvaluation]:
        """
        Use LLM judge to evaluate the response with multi-component scoring.
        Returns a dictionary with individual scores and the judge evaluation.
        """
        judge_evaluation = await self._generate_judge_evaluation(
            question, expected_answer, generated_answer
        )
        return judge_evaluation

    async def _generate_judge_evaluation(
        self, question: str, ground_truth: str, response: str, max_retries: int = 3
    ) -> JudgeEvaluation:
        """
        Generate a judge evaluation with multi-component scoring.
        Parses JSON response from the judge model.
        """
        config = self.config
        responses_create_params = config.judge_responses_create_params.model_copy(deep=True)
        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            question=question, ground_truth=ground_truth, response=response
        )
        responses_create_params.input = [
            NeMoGymEasyInputMessage(
                role="user",
                content=self.JUDGE_SYSTEM_MESSAGE + "\n\n" + judge_prompt,
            ),
        ]

        for attempt in range(max_retries):
            try:
                response_obj = await self.server_client.post(
                    server_name=config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=responses_create_params,
                )
                judge_response = NeMoGymResponse.model_validate(await response_obj.json())
                
                # Extract the text response
                last_output = judge_response.output[-1]
                if last_output.type != "message":
                    continue
                
                last_content = last_output.content[-1]
                if last_content.type != "output_text":
                    continue
                
                output_text = last_content.text
                
                # Try to parse JSON response
                try:
                    result = json.loads(output_text)
                    
                    # Validation of required keys
                    required_keys = [
                        "outcome_correct", "reasoning_coherence", "reasoning_relevance",
                        "format_compliance", "tag_usage", "efficiency"
                    ]
                    if all(k in result for k in required_keys):
                        judge_evaluation = JudgeEvaluation(
                            responses_create_params=responses_create_params,
                            response=judge_response,
                            outcome_correct=float(result["outcome_correct"]),
                            reasoning_coherence=float(result["reasoning_coherence"]),
                            reasoning_relevance=float(result["reasoning_relevance"]),
                            format_compliance=float(result["format_compliance"]),
                            tag_usage=float(result["tag_usage"]),
                            efficiency=float(result["efficiency"])
                        )
                        return judge_evaluation
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract from text
                    pass
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
        
        # Fallback: return default scores
        return JudgeEvaluation(
            responses_create_params=responses_create_params,
            response=judge_response if 'judge_response' in locals() else None,
            outcome_correct=0.0,
            reasoning_coherence=0.0,
            reasoning_relevance=0.0,
            format_compliance=0.0,
            tag_usage=0.0,
            efficiency=0.5
        )


if __name__ == "__main__":
    LibraryJudgeMathResourcesServer.run_webserver()
