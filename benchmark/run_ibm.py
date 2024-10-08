"""An example wrapper script for running experiments on IBM machines.

Relies upon :py:mod:`bench_qubo.DataBaseQUBO` to store instances and
:py:class:`bench_qubo.QUBOSolverIBM` to handle interactions with the (remote)
hardware.
"""
import datetime
import argparse
import os

from bench import __version__ as version
from bench_qubo import DataBaseQUBO, QUBOSolverIBM

def main():
    """Main script code."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str)
    parser.add_argument('--ids', nargs='+', type=str)
    parser.add_argument('--directory', type=str, default=r'..\instances\QUBO')

    args = parser.parse_args()

    benchmark_name = args.name
    allowed_instance_ids = args.ids
    directory = args.directory

    def filter_fun(data):
        return data['description']['instance_id'] in allowed_instance_ids

    def sort_fun(instance_id, data):
        return (len(data['P']), instance_id)

    db = DataBaseQUBO(directory, filter_fun=filter_fun, sort_fun=sort_fun)

    solver = QUBOSolverIBM(db)

    optimizer_kwargs = dict(maxiter=5)
    optimizer_name = 'BOBYQA'
    sampler_optimization_level = 3
    sampler_resilience_level = 1
    estimator_optimization_level = 3
    estimator_resilience_level = 1
    reps_list = [5]
    sampler_shots = 1000
    estimator_shots_list = [100]
    backend_name = 'ibm_cusco'
    service_channel = 'ibm_quantum'
    service_instance = 'ibm-q-fraunhofer/fhg-all/fitw01'
    max_time = '8h'  # 12 h fails
    recover_on_exception = False
    a = 1
    b = 1
    timestamp = datetime.datetime.now().timestamp()

    if not os.path.exists(f'result/{benchmark_name}'):
        os.mkdir(f'result/{benchmark_name}')
    save_file_path = f'result/{benchmark_name}/{benchmark_name}_{backend_name}_{version}_{timestamp}'

    num_repetitions = 1

    run_list = [
        (instance_id, (optimizer_kwargs, optimizer_name, sampler_optimization_level, sampler_resilience_level,
                       estimator_optimization_level, estimator_resilience_level, reps, sampler_shots,
                       estimator_shots, backend_name, service_channel, service_instance, max_time, recover_on_exception, a,
                       b))
        for reps in reps_list for estimator_shots in estimator_shots_list for _ in
        range(num_repetitions) for instance_id in db.get_instance_ids()]

    print(f'{solver.name}: {list(db.get_instance_ids())} -> {save_file_path}')
    solver.benchmark(run_list, save_file_path)


if __name__ == '__main__':
    main()
