# Copyright 2020 Adap GmbH. All Rights Reserved.
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
# ==============================================================================
"""Flower server."""


from .app import start_bo_server as start_bo_server
from flwr.server.client_manager import SimpleClientManager as SimpleClientManager
from flwr.server.history import History as History
from .server import BoServer as BoServer

__all__ = [
    "start_bo_server",
    "SimpleClientManager",
    "History",
    "BoServer",
]
