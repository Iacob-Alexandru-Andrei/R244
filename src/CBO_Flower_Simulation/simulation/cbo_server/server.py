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

import os
from threading import active_count

from flwr.client import client
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
sys.path.append("./src")

from fairbo.utils import opt_adam_max
import torch
from fairbo.models import ExactGPModel
from botorch.utils.sampling import draw_sobol_samples
from fairbo.acquisitions import  FairBatchOWAUCB,  FairBatchOWAUCB_NoPermute
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import datetime as dt




import concurrent.futures
import timeit
from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Reconnect,
    Scalar,
    Weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

DEPRECATION_WARNING_EVALUATE = """
DEPRECATION WARNING: Method

    Server.evaluate(self, rnd: int) -> Optional[
        Tuple[Optional[float], EvaluateResultsAndFailures]
    ]

is deprecated and will be removed in a future release, use

    Server.evaluate_round(self, rnd: int) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]

instead.
"""

DEPRECATION_WARNING_EVALUATE_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_evaluate
return format:

    Strategy.aggregate_evaluate(...) -> Optional[float]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_evaluate(...) -> Tuple[Optional[float], Dict[str, Scalar]]

instead.
"""

DEPRECATION_WARNING_FIT_ROUND = """
DEPRECATION WARNING: The configured Strategy uses a deprecated aggregate_fit
return format:

    Strategy.aggregate_fit(...) -> Optional[Weights]

This format is deprecated and will be removed in a future release. It should use

    Strategy.aggregate_fit(...) -> Tuple[Optional[Weights], Dict[str, Scalar]]

instead.
"""

import copy

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]


class BoServer:
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, num_loops: int = 25, c1 = 0.001, c2 = 1.0, bases = [1.0, 0.2], nAgent = 10, dim_p = 3, vary = False, t0 = 2, seed = [8911, 7, 444, 3525, 5023, 1556, 5399, 7863, 4269, 2973]) -> History:
        """Run federated averaging for a number of rounds."""
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        history = History()
        path ='./results/'
        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        bounds = torch.cat([torch.tensor([[np.log(1e-5), np.log(1e-5), np.log(1e-5)]], dtype=torch.float), torch.tensor([[0, 0., 0.]])])
        hypers = {
            'likelihood.noise_covar.noise': 0.01,
            'covar_module.outputscale': torch.tensor(0.12),
            'covar_module.base_kernel.lengthscale': [1.6, 3.2, 3.6],
            'mean_module.constant': 0.3,
        }
        for base in bases:
            ws = torch.pow(base, torch.arange(nAgent))
            experimentNO = "CNN" + '_base={}_c1={}_vary={}'.format(base, c1, vary)
            with torch.no_grad():
                obsXAll = torch.empty(num_rounds, nAgent,num_loops, dim_p)
                obsYAll = torch.empty(num_rounds, nAgent, num_loops)

            for current_round in range(0, num_rounds):
                l = current_round
                torch.manual_seed(seed[l])
                # Train model and replace previous global model
                
        
                with torch.no_grad():
                    obsX = torch.empty(nAgent, t0, dim_p)
                    obsY = torch.empty(nAgent, t0)
                    obsF = torch.empty(nAgent, t0)

                    for i in range(nAgent):
                        obsX[i] = draw_sobol_samples(bounds=bounds, n=t0, q=1).squeeze_()
                    
                    obsY = self.fit_evaluate_round(rnd = current_round, hyper_params = obsX, t0=t0) #, agentList[i])

                    trainX = obsX.reshape(-1, dim_p)
                    trainY = obsY.reshape(-1)
                
                boModel = ExactGPModel(trainX, trainY)
                hypers = {
                    'likelihood.noise_covar.noise': torch.tensor(0.1).pow(2),
                    'covar_module.outputscale': trainY.mean().clone().detach().requires_grad_(True),
                }
                boModel.initialize(**hypers)
                boModel.set_train_data(trainX, trainY)
                start_time = dt.datetime.now()

                for t in range(t0, num_loops):
                    print(obsY.shape)
                    sorton = torch.sum(obsY, 1)
                    lambd = torch.sum(obsY, 1)

                    ratio = torch.sum(ws)**2 / (torch.sum(ws**2) * nAgent)

                    if vary:
                        c1 = c1*ratio
                    if base < 1:
                        acq_func =  FairBatchOWAUCB(boModel, nAgent, t+1, ws, lambd, C1=c1 , C2=c2)
                    else:
                        acq_func = FairBatchOWAUCB_NoPermute(boModel, nAgent, t+1, ws, lambd, C1=c1, C2=c2)

                    nbounds = torch.cat(nAgent*[bounds], 1)
                    ret = opt_adam_max(acq_func, nbounds, nIter=500, N=20, nSample=1500)
                    del acq_func
                    
                    with torch.no_grad():
                        newX = ret[0].reshape(nAgent, -1).detach()

                        if base < 1:
                            rank = torch.argsort(torch.argsort(sorton))

                            posterior_mean = boModel(newX).mean
                            posterior_argsort = torch.argsort(posterior_mean, descending = True)

                            newX = newX[posterior_argsort]
                            newX = newX[rank]

                    print(torch.exp(newX))
                    newY = self.fit_evaluate_round(rnd = current_round, hyper_params = newX)

                    if t <5:
                        p = 1
                    else:
                        p = 10

                    if t%p == 0:
                        end_time = dt.datetime.now()
                        elapsed_time= (end_time - start_time).seconds/p

                        print('Loop', l+1, ': ', t, ' observations selected.', '  Time per iter: ', elapsed_time, 's')
                        start_time = dt.datetime.now()
                    
                    print(f"ObsX_s:{obsX.shape}, ObsY_y:{obsY.shape}")

                    obsX = torch.cat([obsX, newX.unsqueeze(1)], 1)
                    obsY = torch.cat([obsY, newY.unsqueeze(1)], 1)

                
                    boModel = boModel.get_fantasy_model(newX, newY)
                

                obsXAll[l] = obsX
                obsYAll[l] = obsY

                result = {}
                result['obsX'] = obsXAll
                result['obsY'] = obsYAll

                torch.save(result,  '/home/ubuntu/r244_alex/R244_Project/results/result_'+ experimentNO+'.pt')
        # Bookkeeping
        # end_time = timeit.default_timer()
        # elapsed = end_time - start_time
        # log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate(
        self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        log(WARNING, DEPRECATION_WARNING_EVALUATE)
        res = self.evaluate_round(rnd)
        if res is None:
            return None
        # Deconstruct
        loss, _, results_and_failures = res
        return loss, results_and_failures

    def fit_evaluate_round(
        self, rnd: int, hyper_params, t0 = 1
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate_round: no clients selected, cancel")
            return None
        og = client_instructions[0][1]
        for i in range(1, len(hyper_params)):
            client_instructions[i] =  (client_instructions[i][0], copy.deepcopy(og))
        if t0 > 1:
            for i in range(0, len(hyper_params)):
                for j in range(0, len(hyper_params[i][0])):
                  client_instructions[i][1].config[f"h{j}"] = hyper_params[i][0][j]
                client_instructions[i][1].config[f"seed"] = i % 10
        else:
            for i in range(0, len(hyper_params)):
                for j in range(0, len(hyper_params[i])):
                  client_instructions[i][1].config[f"h{j}"] = hyper_params[i][j]
                client_instructions[i][1].config[f"seed"] = i % 10 
             
        # for i in range(0, len(hyper_params)):
        #     log(DEBUG, f"seed:{client_instructions[i][1].config[f'seed']}" )
        #     log(DEBUG, f"h0:{client_instructions[i][1].config[f'h{0}']}" )
        #     log(DEBUG, f"h1:{client_instructions[i][1].config[f'h{1}']}" )
        #     log(DEBUG, f"h2:{client_instructions[i][1].config[f'h{2}']}" )
          
        # Collect `evaluate` results from all clients participating in this round
        results, failures = fit_evaluate_clients(client_instructions, hyper_params=hyper_params)
        log(
            DEBUG,
            "evaluate_round received %s results and %s failures",
            len(results),
            len(failures),
        )
        
        for i in range(len(results)):
            log( DEBUG,f"index:{i} Id:{  results[i][1].num_examples}")
            
        acc =[ evaluate_res.metrics['accuracy'] for _, evaluate_res in results    ]  
      
        if t0 > 1:
            for z in range(1,t0):
                for i in range(0, len(hyper_params)):
                    for j in range(0, len(hyper_params[i][z])):
                         client_instructions[i][1].config[f"h{j}"] = hyper_params[i][z][j]
                    client_instructions[i][1].config[f"seed"] = i %10

                results, failures = fit_evaluate_clients(client_instructions, hyper_params=hyper_params)
                log(
                    DEBUG,
                    f"t0:{t0} evaluate_round received %s results and %s failures",
                    len(results),
                    len(failures),
                )
                new_acc = [ evaluate_res.metrics['accuracy'] for _, evaluate_res in results    ] 
                acc = list(zip(acc, new_acc))
            acc = torch.tensor(acc)   
        else: 
            acc = torch.tensor(acc)   
        print(acc)
        return acc
  
    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        _ = shutdown(clients=[all_clients[k] for k in all_clients.keys()])

    def _get_initial_parameters(self) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        log(INFO, "Received initial parameters from one random client")
        return parameters_res.parameters


def shutdown(clients: List[ClientProxy]) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(reconnect_client, c, reconnect) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct a single client to disconnect and (optionally) reconnect
    later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect

def fit_evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]], hyper_params
) -> EvaluateResultsAndFailures:
    # for _, ins in client_instructions:
    #     print(ins.config['h1'])
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] =  []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            result = future.result()
            results.append(result)
    return results, failures


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res
