#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

void async_stencil(const int nx, const int ny, float *image, float *tmp_image, int row_size, int world_rank, int world_size, MPI_Request send_request);

void init_image(const int nx, const int ny, float *image, float *tmp_image);

void output_image(const char *file_name, const int nx, const int ny, float *image);

void divide_image(const float *image, float *split_image, float *split_tmp, int world_rank, int split_size, int nx, int ny, int world_size, int row_size);

void gather(int world_rank, float *split_image, float *fin_image, int split_size, int world_size, int row_size, int nx, int ny);

void sendrecv_halo_exchange(float *split_image, int world_rank, int nx, int world_size, int row_size);

void RMA_halo_exchange(float *split_image, int world_rank, int nx, int world_size, int row_size, MPI_Win upper_window, MPI_Win lower_window);

double calc_ncols(int world_rank, int world_size, int nn);

double wtime(void);

MPI_Request send_requests[2];
MPI_Request recv_requests[2];
MPI_Status statuses[2];
int above;
int below;


int main(int argc, char *argv[]) {

    // Check usage
    if (argc != 4) {
        fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size, world_rank, split_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Initialise problem dimensions from command line arguments
    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int niters = atoi(argv[3]);

    // Find the size of the splits
    int row_size = (int) calc_ncols(world_rank, world_size, ny);
    split_size = (nx + 2) * (row_size + 2);

    // Allocate the image space
    float *image = malloc(sizeof(float) * (nx + 2) * (ny + 2));
    float *tmp_image = malloc(sizeof(float) * (nx + 2) * (ny + 2));
    float *split_image = malloc(sizeof(float) * split_size);
    float *split_tmp = malloc(sizeof(float) * split_size);

    // Set the input image
    init_image(nx, ny, image, tmp_image);

    // split the image into equal sized chunks
    divide_image(image, split_image, split_tmp, world_rank, split_size, nx, ny, world_size, row_size);
    MPI_Request send_request;

    double tic = wtime();
    for (int t = 0; t < niters; ++t) {
        async_stencil(nx, ny, split_image, split_tmp, row_size, world_rank, world_size, send_request);
        async_stencil(nx, ny, split_tmp, split_image, row_size, world_rank, world_size, send_request);
    }
    double toc = wtime();

    gather(world_rank, split_image, image, split_size, world_size, row_size, nx, ny);

    if (world_rank == 0) {
        // Output
        printf("------------------------------------\n");
        printf(" runtime: %lf s\n", toc - tic);
        printf("------------------------------------\n");
        output_image(OUTPUT_FILE, nx, ny, image);
        free(image);
    }
    MPI_Finalize();
}

/**
 * Perform asynchronous stencil
 *
 * 1. Send rows before stencil
 * 2. Stencil on inner rows
 * 3. Receive halo
 * 4. wait
 * 5. Stencil on halo rows
 *
 * @param nx
 * @param ny
 * @param image
 * @param tmp_image
 * @param row_size
 * @param world_rank
 * @param world_size
 * @param send_request
 */
void async_stencil(const int nx, const int ny, float *image, float *__restrict__ tmp_image, int row_size, int world_rank,int world_size, MPI_Request send_request) {
    int padding;

    if(world_rank != world_size-1){
        // send bottom halo
        MPI_Isend(image + row_size * ((nx + 2)), nx+2, MPI_FLOAT, world_rank+1, 1, MPI_COMM_WORLD, &send_request);
    }
    if(world_rank != 0){
        // send bottom halo
        MPI_Isend(image + (nx + 2), nx+2, MPI_FLOAT, world_rank-1, 0, MPI_COMM_WORLD, &send_request);
    }

    for (int i = 1; i < row_size - 1; ++i) {
        for (int j = 0; j < nx; ++j) {
            padding = (i * 2) + 3;
            tmp_image[(j + i * nx) + nx + padding] =   image[(j + i * nx) + nx + padding] * (float) 0.6
                                                       + image[((j + i * nx) + nx + padding) + 1] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) - 1] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) + (nx + 2)] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) - (nx + 2)] * (float) 0.1;
        }
    }


    if (world_rank != world_size - 1) {
        // receive bottom halo
        MPI_Recv(image + row_size * ((nx + 2)) + (nx + 2), nx+2, MPI_FLOAT, world_rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (world_rank != 0) {
        // receive top halo
        MPI_Recv(image, nx+2, MPI_FLOAT, world_rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (world_size != 1){
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < nx; ++j) {
            padding = (i * 2) + 3;
            tmp_image[(j + i * nx) + nx + padding] =   image[(j + i * nx) + nx + padding] * (float) 0.6
                                                       + image[((j + i * nx) + nx + padding) + 1] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) - 1] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) + (nx + 2)] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) - (nx + 2)] * (float) 0.1;
        }
    }

    for (int i = row_size - 1; i < row_size; ++i) {
        for (int j = 0; j < nx; ++j) {
            padding = (i * 2) + 3;
            tmp_image[(j + i * nx) + nx + padding] =   image[(j + i * nx) + nx + padding] * (float) 0.6
                                                       + image[((j + i * nx) + nx + padding) + 1] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) - 1] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) + (nx + 2)] * (float) 0.1
                                                       + image[((j + i * nx) + nx + padding) - (nx + 2)] * (float) 0.1;
        }
    }
}

/**
 * Stencil function
 *
 * Not used in this implementation - kept in as proof of work.
 * @param nx
 * @param ny
 * @param image
 * @param tmp_image
 * @param row_size
 * @param world_rank
 */
void stencil(const int nx, const int ny, const float *image, float *__restrict__ tmp_image, int row_size, int world_rank) {
    int padding;
    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < nx; ++j) {
            padding = (i * 2) + 3;
            tmp_image[(j + i * nx) + nx + padding] = image[(j + i * nx) + nx + padding] * (float) 0.6
                                                     + image[((j + i * nx) + nx + padding) + 1] * (float) 0.1
                                                     + image[((j + i * nx) + nx + padding) - 1] * (float) 0.1
                                                     + image[((j + i * nx) + nx + padding) + (nx + 2)] * (float) 0.1
                                                     + image[((j + i * nx) + nx + padding) - (nx + 2)] * (float) 0.1;
        }
    }
}

/***
 * Blocking sendrecv using MPI_SendRecv
 *
 * Not used in this implementation - kept in as proof of work.
 * @param split_image
 * @param world_rank
 * @param nx
 * @param world_size 
 * @param row_size
 */
void sendrecv_halo_exchange(float *split_image, int world_rank, int nx, int world_size, int row_size) {
    // for all except last
    if (world_rank != world_size - 1) {
        MPI_Sendrecv(split_image + row_size * ((nx + 2)), nx + 2, MPI_FLOAT,
                     world_rank + 1, 0, split_image + row_size * ((nx + 2)) + (nx + 2), nx + 2, MPI_FLOAT,
                     world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // for all except first
    if (world_rank != 0) {
        MPI_Sendrecv(split_image + (nx + 2), nx + 2, MPI_FLOAT,
                     world_rank - 1, 0, split_image, nx + 2, MPI_FLOAT,
                     world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/**
 * Remote memory access halo exchange
 *
 * Not used in this implementation - kept in as proof of work.
 * @param split_image
 * @param world_rank
 * @param nx
 * @param world_size
 * @param row_size
 * @param upper_window
 * @param lower_window
 */
void RMA_halo_exchange(float *split_image, int world_rank, int nx, int world_size, int row_size, MPI_Win upper_window, MPI_Win lower_window){
    // create window for halos
    MPI_Win_create(split_image, 1, sizeof(float) * nx+2, MPI_INFO_NULL, MPI_COMM_WORLD, &upper_window);
    MPI_Win_create(split_image + row_size * ((nx + 2)) + (nx + 2), 1, sizeof(float) * nx+2, MPI_INFO_NULL, MPI_COMM_WORLD, &lower_window);
    MPI_Win_fence(0, upper_window);
    MPI_Win_fence(0, lower_window);

    if (world_rank != 0) {
        MPI_Put(split_image + (nx + 2), nx+2, MPI_FLOAT, world_rank -1, 0, nx+2, MPI_FLOAT, lower_window);
    }

    if (world_rank != world_size - 1){
        MPI_Put(split_image + row_size * ((nx + 2)), nx+2, MPI_FLOAT, world_rank + 1, 0, nx+2, MPI_FLOAT, upper_window);
    }

    MPI_Win_fence(0, upper_window);
    MPI_Win_fence(0, lower_window);
}

/**
 * Send all pieces of the grid to master rank (rank 0)
 * @param world_rank 
 * @param split_image 
 * @param fin_image 
 * @param split_size 
 * @param world_size 
 * @param row_size 
 * @param nx 
 * @param ny 
 */
void gather(int world_rank, float *split_image, float *fin_image, int split_size, int world_size, int row_size, int nx, int ny) {
    // All processes sent their chunk to master process
    if (world_rank != 0) {
        MPI_Send(split_image + (nx + 2), split_size - (2 * (nx + 2)), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    // Master process puts sections back together
    if (world_rank == 0) {

        for (int i = 0; i < world_size; i++) {

            // if current process is master process
            if (i == 0) {
                for (int p = 0; p < split_size; p++)
                    fin_image[p] = split_image[p];

                // if current process is not master process
            } else if (i == world_size - 1) {
                int final_rows = split_size + ((ny % world_size) * nx + 2);
                MPI_Recv(fin_image + (row_size * ((nx + 2) * i)) + (nx + 2), final_rows, MPI_FLOAT, i, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(fin_image + (row_size * ((nx + 2) * i)) + (nx + 2), split_size, MPI_FLOAT, i, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

/***
 * Split an input image into equally sized pieces based on world_size
 *
 * @param image 
 * @param split_image 
 * @param split_tmp 
 * @param world_rank 
 * @param split_size 
 * @param nx 
 * @param ny 
 * @param world_size 
 * @param row_size 
 */
void divide_image(const float *image, float *split_image, float *split_tmp, int world_rank, int split_size, int nx, int ny, int world_size, int row_size) {
    int start = (ny / world_size) * ((nx + 2) * world_rank);
    int stop = (ny / world_size) * ((nx + 2) * (world_rank + 1)) + ((2 * (nx) + 3));
    int count = 0;
    int remainder = ny % world_size;

    if (remainder != 0) {
        if (world_rank == world_size - 1) {
            stop += remainder * (nx + 2);
        }
    }
    for (int i = start; i < stop; i++) {
        split_image[count] = image[i];
        split_tmp[count] = 0;
        count++;
    }
}

/***
 * Calculate number of rows, add remainder rows to final rank.
 *
 * @param world_rank 
 * @param world_size 
 * @param nn 
 * @return 
 */

double calc_ncols(int world_rank, int world_size, int nn) {
    int num;
    num = nn / world_size;
    //integer division
    if ((nn % world_size) != 0) {
        if (world_rank == world_size - 1) {
            num += (nn % world_size);
        }
    }
    return num;
}


/***
 * Create the input image
 *
 * @param nx 
 * @param ny 
 * @param image 
 * @param tmp_image 
 */
void init_image(const int nx, const int ny, float *image, float *tmp_image) {
    // Zero everything
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            image[j + i * ny] = 0.0;
            tmp_image[j + i * ny] = 0.0;
        }
    }

    // Checkerboard
    int padding = 3;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int ii = i * ny / 8; ii < (i + 1) * ny / 8; ++ii) {
                for (int jj = (j * nx / 8); jj < (j + 1) * nx / 8; ++jj) {

                    padding = (ii * 2) + 3;
                    int pos = (jj + nx * ii) + nx + padding;


                    if ((i + j) % 2) {
                        // white
                        image[pos] = 100.0;
                    }
                }
            }
        }
    }
}

/**
 * Routine to output the image in Netpbm grayscale binary image format
 *
 * @param file_name 
 * @param nx 
 * @param ny 
 * @param image 
 */
void output_image(const char *file_name, const int nx, const int ny, float *image) {

    // Open output file
    FILE *fp = fopen(file_name, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
        exit(EXIT_FAILURE);
    }

    // Ouptut image header
    fprintf(fp, "P5 %d %d 255\n", nx, ny);

    // Calculate maximum value of image
    // This is used to rescale the values
    // to a range of 0-255 for output
    double maximum = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (image[j + i * ny] > maximum)
                maximum = image[j + i * ny];
        }
    }

    // Output image, converting to numbers 0-255
    int padding = 3;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            int pos = (j + i * nx) + nx + padding;
            fputc((char) (255.0 * image[pos] / maximum), fp);
        }
        padding += 2;
    }
    // Close the file
    fclose(fp);
}

/**
 * Get the current time in seconds since the Epoch
 */
double wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}