import os
import time

import requests

from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum


class TaskType(str, Enum):
    CPU_BOUND = 'cpu_bound'
    NETWORK_BOUND = 'network_bound'
    GPU_BOUND = 'gpu_bound'


class Task(BaseModel):
    typeof: TaskType


app = FastAPI()

DEFAULT_COEFS = {'cpu_bound': {'cpu': 1.0,
                               'network': 0.0,
                               'gpu': 0.0},
                 'network_bound': {'cpu': 0.0,
                                   'network': 1.0,
                                   'gpu': 0.0},
                 'gpu_bound': {'cpu': 0.0,
                               'network': 0.0,
                               'gpu': 1.0}}
TASK_COEFS = {}
WORKER_URLS = ['http://192.168.31.49:8000', 'http://192.168.31.48:8000']
WORKER_STATE: int | None = [None for _ in range(len(WORKER_URLS))]


def init_coefs():
    global TASK_COEFS
    TASK_COEFS.update({t: get_coefs_from_env_vars(t)
                       for t in ['cpu_bound', 'network_bound', 'gpu_bound']})


def get_coefs_from_env_vars(task_type):
    return {t: get_coef(task_type, t)
            for t in ['cpu', 'network', 'gpu']}


def get_coef(task_type, coef_type):
    env_var_name = f'TASK_{task_type.upper()}_{coef_type.upper()}'
    v = os.environ.get(env_var_name)
    return (float(v) if v is not None
            else DEFAULT_COEFS[task_type][coef_type])


def get_task_coefs():
    if not TASK_COEFS:
        init_coefs()

    return TASK_COEFS


def argmax(it):
    _max = None
    max_idx = None
    first = True
    for idx, item in enumerate(it):
        if first:
            _max = item
            max_idx = idx
            first = False
            continue

        if item > _max:
            _max = item
            max_idx = idx

    return max_idx


def fetch_worker_state():
    for idx, url in enumerate(WORKER_URLS):
        WORKER_STATE[idx] = {
            'cpu_perc': 80 + idx*10,
            'gpu_perc': 0,
            'available_RAM': 10000,
            'available_storage': 1000*1000*100
        }
        resp = requests.get(url + '/server/load')
        body_json = resp.json()
        WORKER_STATE[idx] = {'cpu_perc': body_json['available_FLOPS_percentage'],
                             'gpu_perc': 0,
                            'available_RAM': body_json['available_RAM']}


def pick_worker(task_type):
    coefs = get_task_coefs()[task_type]
    fetch_worker_state()
    k_storage = 1
    k_ram = 1
    scores = []
    for idx, w_state in enumerate(WORKER_STATE):
        score =\
            coefs['cpu'] * (w_state['cpu_perc'] / 100) +\
            coefs['gpu'] * (w_state['gpu_perc'] / 100) +\
            coefs['network'] * 1
        score = score * k_storage * k_ram
        scores.append(score)

    worker_idx = argmax(scores)
    return worker_idx


def run_task_in_worker(worker_idx, task_type):
    task_mapping = {'cpu_bound': 'kr1t1ka/cpu-bound',
                    'gpu_bound': 'kr1t1ka/gpu-bound',
                    'network_bound': 'kr1t1ka/network-bound'}
    task_name = task_mapping[task_type]
    url = f'{WORKER_URLS[worker_idx]}/docker/run?image={task_name}&waited=true'
    
    kwargs = {}
    if task_type == 'cpu_bound':
        kwargs['json'] = {'PRECISION': '55000'}
    resp = requests.post(url, **kwargs)
    body_json = resp.json()

    return body_json


@app.post('/api/v1/run')
def make_new_task(task: Task):
    time_run_start = time.time()
    worker_idx = pick_worker(task.typeof.value)
    time_run_task_start = time.time()
    res = run_task_in_worker(worker_idx,'cpu_bound')
    print(f"Time with pick_worker: {time.time() - time_run_start}")
    print(f"Time with out pick_worker: {time.time() - time_run_task_start}")
    print(f"Time real: {res['logs']['execution_time']}")
    return res


@app.post("/api/v1/stop_all")
def stop_all_workers():
    for worker_idx, url in enumerate(WORKER_URLS):
        url = f'{WORKER_URLS[worker_idx]}/docker/containers/all'
        resp = requests.delete(url)
        assert resp.status_code == 204

    return {'status': 'ok'}


@app.get('/api/v1/conf')
def get_conf_handler():
    return get_task_coefs()

