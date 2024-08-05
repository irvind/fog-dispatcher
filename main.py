import os
import requests

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
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
WORKER_URLS = ['http://worker1', 'http://worker2']
WORKER_STATE = [None for _ in range(len(WORKER_URLS))]


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
    for idx, url in enumerate(worker_urls):
        WORKER_STATE[idx] = {
            'cpu_perc': 80 + idx*10,
            'gpu_perc': 0,
            'available_RAM': 10000,
            'available_storage': 1000*1000*100
        }
        # TODO: uncomment
        #resp = requests.get(url + '/server/load')
        #body_json = resp.json()
        #worker_state[idx] = {'cpu_perc': body_json['available_FLOPS_percentage'],
        #                     'available_RAM': body_json['available_RAM']}


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
        score = score * k_storate * k_ram
        scores.append(score)

    worker_idx = argmax(scores)
    return worker_idx


def run_task_in_worker(worker_idx, task_type):
    task_mapping = {'cpu_bound': 'cpu-bound',
                    'gpu_bound': 'gpu-bound',
                    'network_bound': 'network-bound'}
    task_name = task_mappgin[task_type]
    url = f'{WORKER_URLS[worker_idx]}/docker/run?image={task_name}&waited=true'
    
    kwargs = {}
    if task_type == 'cpu_bound':
        kwargs['json'] = {'PRECISION': '55000'}
    resp = requests.post(url, **kwargs)
    body_json = resp.json()

    return body_json


@app.post('/api/v1/run')
def make_new_task(task: Task):
    worker_idx = pick_worker(task.typeof.value)
    return run_task_in_worker(worker_idx)


@app.post("/api/v1/stop_all")
def stop_all_workers():
    for idx, url in enumerate(worker_urls):
        url = f'{worker_urls[worker_idx]}/docker/containers/all'
        resp = requests.delete(url)
        assert resp.status_code == 204

    return {'status': 'ok'}


@app.get('/api/v1/conf')
def get_conf_handler():
    return get_task_coefs()

