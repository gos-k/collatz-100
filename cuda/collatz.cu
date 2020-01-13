#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

const uint64_t max_num = 0x3fffffffffffffff;
const uint64_t table_size = 0x30000000;

static const size_t N = 11;

__device__ uint32_t handled[N] = {0x1249, 0xad2594c3, 0x7ceb0b27, 0x84c4ce0b, 0xf38ace40, 0x8e211a7c, 0xaab24308, 0xa82e8f10, 0x00000000, 0x00000000, 0x00000000};

class BigNum
{
    private:
        uint32_t n[N];

    public:
        __host__ __device__ BigNum()
        {
            for (int i = 0; i < N; i++)
            {
                n[i] = 0;
            }
        }

        __host__ __device__ BigNum(uint32_t x10)
        {
            n[N - 1] = x10;
            for (int i = 0; i < (N - 1); i++)
            {
                n[i] = 0;
            }
        }

        __host__ __device__ BigNum(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t x4, uint32_t x5, uint32_t x6, uint32_t x7, uint32_t x8, uint32_t x9, uint32_t x10)
        {
            n[0] = x0;
            n[1] = x1;
            n[2] = x2;
            n[3] = x3;
            n[4] = x4;
            n[5] = x5;
            n[6] = x6;
            n[7] = x7;
            n[8] = x8;
            n[9] = x9;
            n[10] = x10;
        }

        __host__ __device__ uint32_t operator [] (size_t i) const { return n[i]; }
        __host__ __device__ uint32_t lsw(void) const { return n[N - 1]; } 

        __host__ __device__ void shift_left(void)
        {
            for (size_t i = 0; i < (N - 1); i++)
            {
                n[i] = (n[i] << 1) | ((n[i + 1] >> 31) & 1);
            }
            n[N - 1] <<= 1;
        }

        __host__ __device__ void shift_right(void)
        {
            for (size_t i = (N - 1); i > 0; i--)
            {
                n[i] = (n[i] >> 1) | ((n[i - 1] << 31) & 0x80000000);
            }
            n[0] >>= 1;
        }

        __host__ __device__ void add(const BigNum & rhs)
        {
            uint64_t x = 0;
            for (int i = (N - 1); i >= 0; i--)
            {
                uint64_t y = (uint64_t)n[i] + (uint64_t)rhs[i] + x;
                n[i] = y & 0xffffffff;
                x = (y >> 32) & 0xffffffff;
            }
        }

        __host__ __device__ bool eq(const BigNum & rhs) const
        {
            for (size_t i = 0; i < N; i++)
            {
                if (n[i] != rhs[i]) { return false; }
            }

            return true;
        }

        __host__ __device__ bool ne(const BigNum & rhs) const
        {
            return !eq(rhs);
        }

        __host__ __device__ bool lt(const BigNum & rhs) const 
        {
            for (size_t i = 0; i < N; i++)
            {
                if (n[i] < rhs[i]) { return true; }
            }

            return false;
        }

        void print(char c = '\0')
        {
            printf("0x");
            for (size_t i = 0; i < N; i++)
            {
                printf("%08x", n[i]);
                if (c != '\0') { printf("%c", c); }
            }
            printf("\n");
        }
};

__host__ __device__ uint32_t lsw(uint32_t * n)
{
    return n[N - 1];
} 

__host__ __device__ void shift_left(uint32_t * n)
{
    for (size_t i = 0; i < (N - 1); i++)
    {
        n[i] = (n[i] << 1) | ((n[i + 1] >> 31) & 1);
    }
    n[N - 1] <<= 1;
}

__host__ __device__ void shift_right(uint32_t * n)
{
    for (size_t i = (N - 1); i > 0; i--)
    {
        n[i] = (n[i] >> 1) | ((n[i - 1] << 31) & 0x80000000);
    }
    n[0] >>= 1;
}

__host__ __device__ int collatz(uint64_t n)
{
    int m = 0;

    if (n < 2) { return 0; }
    while (n != 1)
    {
        if (n > max_num)
        {
            m = 0;
            break;
        }

        n = (n & 1) ? ((3 * n + 1) / 2) : (n / 2);
        m += (n & 1) ? 2 : 1;
    }

    return m;
}

__host__ __device__ void add(uint32_t * n, uint32_t * rhs)
{
    uint64_t x = 0;
    for (int i = (N - 1); i >= 0; i--)
    {
        uint64_t y = (uint64_t)n[i] + (uint64_t)rhs[i] + x;
        n[i] = y & 0xffffffff;
        x = (y >> 32) & 0xffffffff;
    }
}

__host__ __device__ void inc(uint32_t * n)
{
    uint64_t z = (uint64_t)n[N - 1] + 1;
    n[N - 1] = z & 0xffffffff;
    uint64_t x = (z >> 32) & 0xffffffff;
    for (int i = (N - 2); i >= 0; i--)
    {
        uint64_t y = (uint64_t)n[i] + x;
        n[i] = y & 0xffffffff;
        x = (y >> 32) & 0xffffffff;
    }
}

__host__ __device__ void inc2(uint32_t * n)
{
    uint64_t z = (uint64_t)n[N - 1] + 2;
    n[N - 1] = z & 0xffffffff;
    uint64_t x = (z >> 32) & 0xffffffff;
    for (int i = (N - 2); i >= 0; i--)
    {
        uint64_t y = (uint64_t)n[i] + x;
        n[i] = y & 0xffffffff;
        x = (y >> 32) & 0xffffffff;
    }
}

__host__ __device__ void add1(uint32_t * lhs, uint32_t * rhs)
{
    uint32_t carry = 1;
    for (int i = (N - 1); i >= 0; i--)
    {
        uint32_t x = lhs[i];
        uint32_t y = rhs[i];
        uint32_t z = x + y + carry;
        carry = ((x & y) | (x ^ y & z)) >> 31;

        lhs[i] = z;
    }
}

#if 0
__host__ __device__ void add1(uint32_t * n, uint32_t * rhs)
{
    uint64_t z = (uint64_t)n[N - 1] + (uint64_t)rhs[N - 1] + 1;
    n[N - 1] = z & 0xffffffff;
    uint64_t x = (z >> 32) & 0xffffffff;

    for (int i = (N - 2); i >= 0; i--)
    {
        uint64_t y = (uint64_t)n[i] + (uint64_t)rhs[i] + x;
        n[i] = y & 0xffffffff;
        x = (y >> 32) & 0xffffffff;
    }
}
#endif

__host__ __device__ void shift_left_add1(uint32_t * lhs, uint32_t * rhs)
{
    uint32_t carry = 0;
    for (int i = (N - 1); i >= 0; i--)
    {
        uint32_t x = lhs[i];
        uint32_t z = (x << 1) | carry;
        carry = x >> 31;

        lhs[i] = z;
    }

    carry = 1;
    for (int i = (N - 1); i >= 0; i--)
    {
        uint32_t x = lhs[i];
        uint32_t y = rhs[i];
        uint32_t z = x + y + carry;
        carry = ((x & y) | (x ^ y & z)) >> 31;

        lhs[i] = z;
    }
}

#if 0
__host__ __device__ void shift_left_add1(uint32_t * lhs)
{
    uint32_t carry = 1;
    uint32_t carry0 = 0;
    for (int i = (N - 1); i >= 0; i--)
    {
        uint32_t x0 = lhs[i];
        uint32_t z0 = (x0 << 1) | carry0;
        carry0 = x0 >> 31;

        uint32_t x = z0;
        uint32_t y = x0;
        uint32_t z = x + y + carry;
        carry = ((x & y) | (x ^ y & z)) >> 31;

        lhs[i] = z;
    }
}
#endif

__host__ __device__ bool is_zero(uint32_t * n)
{
    uint32_t x = 0;
    for (int i = 0; i < N; i++)
    {
        x |= n[i];
    }

    return (x == 0) ? true : false;
}

__host__ __device__ bool is_one(uint32_t * n)
{
    uint32_t x = n[0];
    for (int i = 1; i < (N - 1); i++)
    {
        x |= n[i];
    }

    return ((x == 0) && (n[N - 1] == 1)) ? true : false;
}

__host__ __device__ bool lt(uint32_t * n, uint32_t rhs)
{
    uint32_t x = n[0];
    for (int i = 1; i < (N - 1); i++)
    {
        x |= n[i];
    }

    return ((x == 0) && (n[N - 1] < rhs)) ? true : false;
}

__host__ __device__ bool less_than(uint32_t * n, uint32_t * rhs)
{
    for (size_t i = 0; i < N; i++)
    {
        if (n[i] < rhs[i]) { return true; }
        if (n[i] > rhs[i]) { return false; }
    }

    return false;
}

__host__ __device__ int collatz_big_num(uint32_t * x, uint16_t * d_t, uint64_t table_size)
{
    int c = 0;

    if (is_zero(x)) { return 0; }
    uint32_t n[N];
    memcpy(n, x, sizeof(n));

    while (!is_one(n))
    {
        if (lsw(n) & 1)
        {
            uint32_t m[N];
            memcpy(m, n, sizeof(m));
            //shift_left_add1(n, m);
#if 1
            shift_left(n);
            add1(n, m);
#endif
            ++c;
        }

        ++c;
        shift_right(n);

#if 0
        if (lt(n, table_size)) {
            c += d_t[lsw(n)];
            break;
        }
#endif
    }

    return c;
}

__host__ __device__ int collatz(BigNum n)
{
    const BigNum one(1);
    int m = 0;

    if (!one.lt(n)) { return 0; }
    while (n.ne(one))
    {
        if ((n.lsw() % 2) == 0)
        {
            n.shift_right();
        }
        else
        {
            BigNum m = n;
            n.shift_left();
            n.add(m);
            n.add(one);
        }
        ++m;
    }

    return m;
}

#define ODD(n)  (3 * (n) + 1)
#define EVEN(n)  ((n) / 2)

__host__ __device__ int collatz_with_table(uint64_t n, uint64_t ts, uint16_t * t)
{
    int m = 0;

    if (n < 2) { return 0; }
    while (n != 1)
    {
        if (n > max_num)
        {
            m = 0;
            break;
        }

        int flag = n & 1;
        n = flag ? EVEN(ODD(n)) : EVEN(n);
        m += flag ? 2 : 1;

        if (n < ts) {
            m += t[n];
            break;
        }
    }

    return m;
}

__global__ void generate_table_cuda(uint16_t * d_table, uint64_t table_size)
{
    int s = gridDim.x * blockDim.x;
    int n = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = n; i < table_size; i += s) {
        d_table[i] = collatz(i);
    }
}

void generate_table(uint16_t * table, uint64_t table_size)
{
#if 1
    puts("generate_table() start.");
    uint16_t * d_table;
    cudaMalloc((void **)&d_table, sizeof(uint16_t) * table_size);
    cudaFuncSetCacheConfig(generate_table_cuda, cudaFuncCachePreferL1);
    generate_table_cuda<<<128, 1024>>>(d_table, table_size);
    cudaMemcpy(table, d_table, sizeof(uint16_t) * table_size, cudaMemcpyDeviceToHost);
    cudaFree(d_table);
    puts("generate_table() finish.");
#else
    puts("generate_table() start.");
    table[0] = table[1] = 0;
    for (uint64_t i = 2; i < table_size; i++)
    {
        //table[i] = collatz_with_table(i, i - 1, table);
        table[i] = collatz(i);
    }
    puts("generate_table() finish.");
#endif
}

#define RAND (r = r * 134775813 + 1)

__host__ __device__ void longest_random(uint32_t * d_z, uint16_t * d_t, uint64_t table_size, uint32_t end, uint32_t seed, uint32_t x)
{
    uint32_t i = x * N;
    uint32_t r = seed + x;

    int max = collatz_big_num(&d_z[i], d_t, table_size);
    for (uint32_t j = 0; j < end; j++)
    {
        uint32_t n[N];
        n[0] = RAND & 0x1fff;
        n[1] = RAND;
        n[2] = RAND;
        n[3] = RAND;
        n[4] = RAND;
        n[5] = RAND;
        n[6] = RAND;
        n[7] = RAND;
        n[8] = RAND;
        n[9] = RAND;
        n[10] = RAND | 1;
        if (!less_than(n, handled)) { continue; }

        int y = collatz_big_num(n, d_t, table_size);
        if (y > max)
        {
            max = y;
            memcpy(&d_z[i], n, sizeof(uint32_t) * N);
        }
    }
}

__global__ void longest_random_cuda(uint32_t * d_z, uint16_t * d_t, uint64_t table_size, uint32_t end, uint32_t seed)
{
    uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
    longest_random(d_z, d_t, table_size, end, seed, x);
}

__global__ void longest_divide_cuda(uint32_t * d_z, uint16_t * d_t, uint64_t table_size, uint64_t start, uint64_t end, uint32_t alfa)
{
    uint64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t y = gridDim.x * blockDim.x;

    uint32_t k[N];
    uint32_t n[N];
    memcpy(k, &d_z[x * N], sizeof(k));
    memcpy(n, k, sizeof(k));
    int max = is_zero(k) ? 0 : collatz_big_num(n, d_t, table_size);
    for (uint64_t j = x + start; j < end; j += y)
    {
        n[alfa] = (uint32_t)j;
        if (!less_than(n, handled)) { break; }

        int y = collatz_big_num(n, d_t, table_size);
        if (y > max)
        {
            max = y;
            memcpy(k, n, sizeof(k));
        }
    }

    memcpy(&d_z[x * N], k, sizeof(k));
}
#if 0
__global__ void longest_divide_cuda(uint32_t * d_z, uint16_t * d_t, uint64_t table_size, uint64_t start, uint64_t end, uint32_t alfa)
{
    uint64_t x = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t y = gridDim.x * blockDim.x;

    uint32_t k[N];
    memcpy(k, &d_z[x * N], sizeof(k));
    int max = is_zero(k) ? 0 : collatz_big_num(k, d_t, table_size);
    for (uint64_t j = 2 * x + 1 + start; j < end; j += y)
    {
        uint32_t n[N];
        for (int i = 0; i < N; i++)
        {
            n[i] = (alfa == i) ? (uint32_t)j : 0;
        }
        if (!less_than(n, handled)) { break; }

        int y = collatz_big_num(n, d_t, table_size);
        if (y > max)
        {
            max = y;
            memcpy(k, n, sizeof(k));
        }
    }

    memcpy(&d_z[x * N], k, sizeof(k));
}
#endif

__global__ void local_search_cuda(uint32_t * d_z, uint16_t * d_t, uint32_t table_size, uint32_t end)
{
    uint64_t x = blockDim.x * blockIdx.x + threadIdx.x;

    uint32_t k[N];
    uint32_t n[N];
    for (int i = 0; i < N; i++)
    {
        n[i] = k[i] = d_z[x * N + i];
    }

    int max = collatz_big_num(k, d_t, table_size);
    for (uint32_t j = 0; j < end; j++)
    {
        inc2(n);
        if (!less_than(n, handled)) { break; }
        int y = collatz_big_num(n, d_t, table_size);

        if (y > max)
        {
            max = y;
            for (int i = 0; i < N; i++)
            {
                k[i] = n[i];
            }
        }
    }

    for (int i = 0; i < N; i++)
    {
        d_z[x * N + i] = k[i];
    }
}

BigNum longest_host(uint32_t * z, int threads)
{
    int max = 0;
    BigNum max_num;
    for (int j = 0; j < threads; j++)
    {
        size_t index = j * N;
        BigNum bn(z[index + 0], z[index + 1], z[index + 2], z[index + 3], z[index + 4], z[index + 5], z[index + 6], z[index + 7], z[index + 8], z[index + 9], z[index + 10]);
        int y = collatz(bn);

        if (y > max)
        {
            max = y;
            max_num = bn;
        }
        break;
    } 
    max_num.print();
    printf("%d\n", max);

#if 0
    for (int j = 0; j < threads; j++)
    {
        size_t index = j * N;
        for (int i = 0; i < N; i++)
        {
            z[index + i] = max_num[i];
        }
    }
#endif

    return max_num;
}

void copy(uint32_t * z, int threads, BigNum max_num)
{
    for (int j = 0; j < threads; j++)
    {
        size_t index = j * N;
        for (int i = 0; i < N; i++)
        {
            z[index + i] = max_num[i];
        }
    }
}

//void longest(uint16_t * table, uint64_t table_size)
void longest(uint16_t * table, uint64_t table_size, int grid_dim, int block_dim)
{
    puts("longest() start.");
    //const int grid_dim = 128;
    //const int block_dim = 128;
    const int threads = grid_dim * block_dim;
    uint32_t z[threads][N];
    uint32_t * d_z;
    uint16_t * d_t;

    cudaMalloc((void **)&d_z, sizeof(z));
    cudaMalloc((void **)&d_t, sizeof(uint16_t) * table_size);
    cudaMemcpy(d_t, table, sizeof(uint16_t) * table_size, cudaMemcpyHostToDevice);

    BigNum max_bn;
    int max_len = 0;

#if 0
    cudaMemset(d_z, 0, sizeof(z));
    //for (int k = 0; k < 100; k++)
    int k = 0;
    {
        //for(uint32_t i = 0; i < N; i++)
        uint32_t i = 1;
        {
            cudaFuncSetCacheConfig(longest_divide_cuda, cudaFuncCachePreferL1);
            //uint64_t total = 0x100000000;
            uint64_t total = 0x40000000;
            uint64_t step = 0x100000;
            uint64_t loop = total / step;
            for (uint64_t j = 0; j < loop; j++)
            {
                longest_divide_cuda<<<grid_dim, block_dim>>>(d_z, d_t, table_size, step * j, step * (j + 1), i);
            }

            cudaMemcpy(z, d_z, sizeof(z), cudaMemcpyDeviceToHost);
            printf("%u, %u, ", k, i);
            longest_host((uint32_t *)z, threads);
            //copy((uint32_t *)z, threads, bn);
            cudaMemcpy(d_z, z, sizeof(z), cudaMemcpyHostToDevice);
            fflush(stdout);
            fflush(stderr);

#if 0
            int y = collatz(bn);
            if (max_len < y)
            {
                max_len = y;
                max_bn = bn;
                break;
            }
#endif
        }
    }
#endif

    srand(time(0));
    cudaMemset(d_z, 0, sizeof(z));
    int count = 0;
    for (uint64_t i = 0; i < 5000000; i++)
    //for (uint64_t i = 0; i < 10; i++)
    {
        longest_random_cuda<<<grid_dim, block_dim>>>(d_z, d_t, table_size, 100, (uint32_t)rand());
        cudaThreadSynchronize();
        //local_search_cuda<<<grid_dim, block_dim>>>(d_z, d_t, table_size, 1000);
        //cudaThreadSynchronize();

        if ((count++ % 0x100) == 0)
        {
            cudaMemcpy(z, d_z, sizeof(z), cudaMemcpyDeviceToHost);

            //printf("%lu\n", i);
            longest_host((uint32_t *)z, threads);
            fflush(stdout);
            fflush(stderr);
        }
    }
    cudaFree(d_z);

    puts("longest() finish.");
}

int main(int argc, char * argv[])
{
    static uint16_t table[table_size];
    generate_table(table, table_size);
    longest(table, table_size, 128, 256);

    return 0;
}
