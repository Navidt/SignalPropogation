#include <iostream>
#include <random>
#include <time.h>

int main() {
  srand(time(NULL));

  std::uniform_real_distribution<double> unif(0,1);
  std::default_random_engine re;
  double a_random_double = unif(re);
  char c = a_random_double << 56;
  // char c = rand();
  double k = (int) c;
  printf("%f\n", k);
}