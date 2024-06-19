import os
import jax

# Get the current XLA_FLAGS
xla_flags = os.environ.get('XLA_FLAGS', '')

# Split the flags into a list
flags = xla_flags.split()

# Define the flags to remove
flags_to_remove = ['--xla_gpu_enable_async_all_gather=true', 
                   '--xla_gpu_enable_async_reduce_scatter=true', 
                   '--xla_gpu_enable_triton_gemm=false']

# Remove the flags
for flag in flags_to_remove:
    if flag in flags:
        flags.remove(flag)

# Join the flags back into a string
xla_flags = ' '.join(flags)

# Set the XLA_FLAGS environment variable
os.environ['XLA_FLAGS'] = xla_flags

print(jax.devices())
# import jax; print(jax.devices())