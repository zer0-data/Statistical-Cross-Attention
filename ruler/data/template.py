# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

PROMPT_TEMPLATES = {
    'base': {"template": "{task_template}", "stop_words": ["<|end_of_text|>"]},
    'llama3': {
        "template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "stop_words": ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"],
    },
    'qwen3': {
        "template": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{task_template}<|im_end|>\n<|im_start|>assistant\n",
        "stop_words": ["<|im_end|>", "<|endoftext|>"],
    },
}
