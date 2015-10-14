__kernel void add(__global const float * a,
                  __global const float * b,
                  const unsigned int n,
                  __global float * c) {
    int gid = get_global_id(0);

    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}
