import sm_openmpi

sm_openmpi.sm_openmpi('scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py', [
            '--num_batches=1000', 
            '--model', 'vgg16', 
            '--batch_size', '64', 
            '--variable_update', 'horovod', 
            '--horovod_device', 'gpu', 
            '--use_fp16'])