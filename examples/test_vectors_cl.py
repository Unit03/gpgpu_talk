import numpy
import os
import pyopencl


ARRAY_SIZE = 4096


def add(a, b):
    # Create context.
    context = pyopencl.create_some_context()

    # Create command queue withing it.
    queue = pyopencl.CommandQueue(context)

    # Build the "program".
    program = pyopencl.Program(
        context,
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "vectors_cl.cl")
        ).read()
    ).build()

    # Create two readable buffers on the device memory and copy the input data
    # there.
    a_in = pyopencl.Buffer(
        context,
        pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
        hostbuf=a)
    b_in = pyopencl.Buffer(
        context,
        pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
        hostbuf=b)

    # Create one writeable buffer on the device memory for result.
    c_out = pyopencl.Buffer(
        context,
        pyopencl.mem_flags.WRITE_ONLY,
        a.nbytes  # Size.
        )

    # Execute the kernel.
    program.add(queue, a.shape, None, a_in, b_in, numpy.uint32(ARRAY_SIZE),
                c_out)

    # Create empty numpy array on the host for result.
    c = numpy.empty_like(a)

    # Copy the result from the device to the host.
    pyopencl.enqueue_copy(queue, c, c_out)

    return c


def print_arrays(a, b, c):
    # Prints just like C version.
    for j in (0, 1, int(a.size / 4 - 2), int(a.size / 4 - 1)):
        print("\t".join(["{:.0f} + {:.0f} = {:.0f}".format(a[i], b[i], c[i])
                         for i in range(j * 4, (j + 1) * 4)]))
        if j == 1:
            print("...")


def test_add():
    # Generate the input array on the host.
    a = numpy.empty(ARRAY_SIZE, dtype=numpy.float32)
    b = numpy.empty(ARRAY_SIZE, dtype=numpy.float32)

    for i in range(ARRAY_SIZE):
        a[i] = i
        b[i] = 2 * i

    c = add(a, b)

    print_arrays(a, b, c)

    assert c[0] == 0
    assert c[1] == 3
    assert c[-2] == 12282
    assert c[-1] == 12285


if __name__ == "__main__":
    test_add()
