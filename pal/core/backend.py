# Copyright 2022 PAL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
import time
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

# GPT-3 API
def call_gpt(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, majority_at=None):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 2
    
    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            ans = openai.Completion.create(
                            model=model,
                            max_tokens=max_tokens,
                            stop=stop,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            n=requested_completions,
                            best_of=requested_completions)
            completions.extend([choice['text'] for choice in ans['choices']])
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.error.RateLimitError as e:
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')
