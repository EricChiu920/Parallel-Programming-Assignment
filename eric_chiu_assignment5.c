// Title : Eric_chiu_Assignment5
// Author : Eric Chiu
// Created on : December 3, 2017
// Description : C program using MPI for parallel computing
// Purpose : Assignment 5 for 493.65 Parallel computing
//           solves the parking garage problem
// Usage : Run with command line arguments: num_stalls exp_exp_mean gauss_exp_mean
//         where num_stalls is the number of parking spaces available in the garage,
//         exp_exp_mean is the exponential distribution of the arrival times of the cars in minutes,
//         and gauss_exp_mean is exp_mean of a normal distribution that determines how long a car stays in a parking spot
// Build with : mpicc eric_chiu_assignment5.c -o eric_chiu_assignment5.exe -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define ARGUMENT_ERROR  1
#define MALLOC_ERROR    2

int handle_argument_errors(int argc, char* argv[]);
double gen_number( double lambda );
double accept_reject(double gauss_mean);
void check_error(int error);

int main (int argc, char* argv[]) {
  int id, p;
  int i, j;
  int num_stalls;
  int num_cars = 100000;
  int converged = 0;
  int error = 0;
  unsigned int max_runs = 100000000;  //Run at most 10^8 iterations
  double rejected = 0, rejected_total = 0, rejected_percent = -1, rejected_temp_percent;
  double average_cars = -1, occupied_stalls, occupied_stalls_total, average_cars_temp_percent;
  double *garage;
  double exp_mean, gauss_mean;
  double x;
  double total_time = 0;
  double count = 0;         //Keep track of the total number of cars
  double tolerance = 0.001; //Keep running until change is less than 10^-3 or reached max_runs
  double p_max_time;        /*For timing*/
  double p_total_time;      /*For timing*/
  double seconds;           /*For timing*/
  double  lambda;            /* parameter of exponential             */
  char *p_end;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  //Start timing for program
  MPI_Barrier (MPI_COMM_WORLD);
  seconds = -MPI_Wtime();

  //Initialize program and check for errors ===================================

  if(id == 0) {
    error = handle_argument_errors(argc, argv);
  }
  MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  check_error(error);

  //get arguments
  num_stalls  = strtol(argv[1], &p_end, 10);
  exp_mean = strtol(argv[2], &p_end, 10);
  gauss_mean   = strtol(argv[3], &p_end, 10);

  if(id == 0) {
    if(num_stalls == 0) {
      error = ARGUMENT_ERROR;
      fprintf(stderr, "Error setting num_stalls.\n");
    }
    else if(exp_mean == 0) {
      error = ARGUMENT_ERROR;
      fprintf(stderr, "Error setting exp_mean\n");
    }
    else if(gauss_mean == 0) {
      error = ARGUMENT_ERROR;
      fprintf(stderr, "Error setting gauss_mean\n");
    }
  }
  MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  check_error(error);
  //end get arguments

  //Allocate memory for the number of stalls in garage
  garage = (double*) malloc(sizeof(double) * num_stalls);
  if(garage == NULL) {
    printf("Could not alloc memory\n");
  }
  MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  check_error(error);

  //End initialize program and check for errors ================================

  //initialize all stalls in garage to 0
  for(i = 0; i < num_stalls; i++){
    garage[i] = 0;
  }

  /* Invert exp_mean to get exponential parameter*/
  lambda = 1.0 / exp_mean;

  /* Seed the random number generator with current time */
  srandom(time(NULL) + id);

  //Simulate a garage
	while(1) {
    //Each processor runs 1,000,000 car simulations and keeps count of rejected cars.
		for(i = 0; i < num_cars; i++) {
      //generate a car arrival time
			x = gen_number(lambda);
	    total_time += x;

      //check if there is a open still in the garage
      for(j = 0; j < num_stalls; j++){
        if(garage[j] < total_time) {
          garage[j] = total_time + accept_reject(gauss_mean);
          break;
        }
        occupied_stalls++; //Count number of occupied stalls
      } //end j loop
      //Count the rejected cars if all the stalls are full
      if(j == num_stalls) {
        rejected++;
      }
			count++; //Keep track of the total number of cars
    } //end i loop

    //Perform reduction on all rejected cars to quickly find a convergence
	  MPI_Allreduce(&rejected, &rejected_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	  MPI_Allreduce(&occupied_stalls, &occupied_stalls_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //Calculate the percentage of rejected and average cars
		rejected_temp_percent = rejected_total / count / p;
    average_cars_temp_percent = occupied_stalls_total / count / p;

    //Break out of loop if coverged or reached total number of runs.
    if((fabs(rejected_percent - rejected_temp_percent) < tolerance && fabs(average_cars - average_cars_temp_percent) < tolerance)
    || rejected_total >= max_runs) {
			break;
		}

    //If it doesn't converge get the new percent.
    rejected_percent = rejected_temp_percent;
    average_cars = average_cars_temp_percent;
	} //End simulate a garage


  MPI_Barrier(MPI_COMM_WORLD);
  seconds += MPI_Wtime();
  p_max_time = seconds;

  MPI_Reduce ( &seconds, &p_max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce ( &seconds, &p_total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(id == 0 && !(count >= max_runs)) {
    printf("\n");
    printf("Convergence for average percent of rejected cars is: %2.3f%%\n", rejected_temp_percent * 100);
    printf("Convergence for average percent of occupied stalls is: %2.3f cars\n", average_cars);
  } else if (id == 0) {
    printf("Could not find a convergence. Ending after 100,000,000 iterations\n");
  }

  if(id == 0) {
    fprintf (stderr, "Running on %d processors. Elapsed time %f seconds. Total time %f  seconds\n", p, p_max_time, p_total_time);
  }

  free(garage);
  MPI_Finalize();
  return 0;
}

int handle_argument_errors(int argc, char* argv[]) {
  if(argc < 4) {
    fprintf(stderr, "Incorrect number of arguments\n");
    return 1;
  }
  if(argc > 4) {
    fprintf(stderr, "Warning: ignoring arguments after the third one.\n");
    return 1;
  }
  return 0;
}

double gen_number( double lambda )
{
    double u = (double) (random()) / RAND_MAX;
    return ( - log ( u ) / lambda );
}

double accept_reject(double gauss_mean) {
  double u1, z, accept;

  //Acceptance rejection for the standard normal deviation
  do {
    u1 = (double) random()/RAND_MAX;    //Generate a Random U1 from U(0,1)
    z = gen_number(1);                  //Generate a Z with mean 1 using inverse CDF
    accept = exp(-0.5*pow((z - 1),2) ); //Test acceptance
  } while(u1 > accept);                 //Repeat until u1 is less than or equal to accept

	//scale number by the standard deviation
	z = z * gauss_mean / 4;

	//Half the time return the negative value
	if((double) random()/RAND_MAX < .5) {
		z = 0 - z;
	}

  //change the mean to gauss_mean
  z = z + gauss_mean;

  return z;
}

void check_error(int error) {
  //exit if errors occured
  if(error != 0) {
    fprintf(stderr, "Program terminating with error code %d\n", error);
    MPI_Finalize();
    exit(error);
  }
}
