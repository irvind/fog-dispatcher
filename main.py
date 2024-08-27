import os
import time
import random

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


class InTaskCoefs(BaseModel):
    typeof: TaskType
    cpu: float
    gpu: float
    network: float


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
WORKER1_URL = 'http://192.168.31.49:8000'
WORKER2_URL = 'http://192.168.31.48:8000'

TASK_COEFS = {}
WORKER_URLS = [WORKER1_URL, WORKER2_URL]
WORKER_STATE: int | None = [None for _ in range(len(WORKER_URLS))]
WORKER_NAME_MAP = {'http://192.168.31.49:8000': 'first (49)', 'http://192.168.31.48:8000': 'second (48)'} 


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


def fetch_worker_state() -> None:
    """
    Получаем информацию о текущей нагрузке воркеров и сохраняем её в памяти.
    """
    for idx, url in enumerate(WORKER_URLS):
        resp = requests.get(url + '/server/load')
        body_json = resp.json()
        # print(body_json)
        WORKER_STATE[idx] = {'cpu_perc': body_json['available_FLOPS_percentage'],
                             'gpu_perc': body_json['available_gpu_FLOPS_percentage'],
                             # TODO: замокать по другому
                             'net_delay': random.random(),
                             'available_RAM': body_json['available_RAM']}
        print(f"WORKER_STATE {url}: {WORKER_STATE[idx]}")


def pick_worker(task_type: str) -> int:
    """
    Возвращаем индекс воркера, на котором будем выполнять задачу.
    """
    coefs = get_task_coefs()[task_type]
    fetch_worker_state()

    k_storage, k_ram = 1, 1
    scores = []
    for idx, w_state in enumerate(WORKER_STATE):
        score =\
            coefs['cpu'] * (w_state['cpu_perc'] / 100) +\
            coefs['gpu'] * (w_state['gpu_perc'] / 100) +\
            coefs['network'] * (w_state['net_delay'])
        score = score * k_storage * k_ram
        scores.append(score)
        print(f'Worker score "{WORKER_URLS[idx]}": {score}')

    worker_idx = argmax(scores)

    return worker_idx


def run_task_in_worker(worker_idx: int, task_type: str) -> dict: 
    """
    Запускаем задачу на конкретном воркере и возвращаем тело ответа.
    """
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
    task_res = run_task_in_worker(worker_idx,'cpu_bound')

    pick_worker_time = time.time() - time_run_start
    without_pick_worker_time = time.time() - time_run_task_start
    real_time = task_res['logs']['execution_time']

    meta_resp = {'pick_worker_time': pick_worker_time,
                 'without_pick_worker_time': without_pick_worker_time,
                 'real_time': real_time,
                 'worker_name': WORKER_NAME_MAP[WORKER_URLS[worker_idx]]}
    
    #print(f"Time with pick_worker: {pick}")
    #print(f"Time with out pick_worker: {time.time() - time_run_task_start}")
    #print(f"Time real: {res['logs']['execution_time']}")
    print(meta_resp)
    return meta_resp


@app.post("/api/v1/stop_all")
def stop_all_workers():
    for worker_idx, url in enumerate(WORKER_URLS):
        url = f'{WORKER_URLS[worker_idx]}/docker/containers/all'
        resp = requests.delete(url)
        assert resp.status_code == 204

    return {'status': 'ok'}


@app.get('/api/v1/coefs')
def get_coefs_handler():
    return get_task_coefs()


@app.post('/api/v1/coefs')
def set_coefs_handler(coefs: InTaskCoefs):
    TASK_COEFS[coefs.typeof] = {'cpu': coefs.cpu,
                                'gpu': coefs.gpu,
                                'network': coefs.network}
    return {'status': 'ok'}

