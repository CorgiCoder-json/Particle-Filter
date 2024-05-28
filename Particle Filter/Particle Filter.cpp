//***************************************************************************
//
// Programmer: Jacob Maurer
// Date: 5/14/2024
// Description: Implementation of a particle filter from this book
// https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
//
//***************************************************************************

#include <iostream>

#include <Eigen/Dense>
#include<array>
#include <Eigen/src/Core/Matrix.h>
#include <cmath>
#include <random>

using namespace std;

const double pi = 3.1415926535897932384626;

//Begginning of the Particle Filter Class
template<int state_size, int number, int measure_size>
class ParticleFilter
{
public:
	//Construct a particle filter with a starting mean and starting std_
	ParticleFilter(Eigen::Vector<double, state_size> mean, Eigen::Vector<double, state_size> std_)
	{
		create_gaussian(mean, std_);
		weights.fill(1.0/((double) number));
		state_mean = mean;
		state_var = std_;
	}

	//Makes particles that are normally distributed
	void create_gaussian(Eigen::Vector<double, state_size> mean, Eigen::Vector<double, state_size> std_)
	{
		
		for (int col = 0; col < state_size; col++)
		{
			normal_distribution<double> distribution(mean(col), std_(col));
			for (int row = 0; row < number; row++)
			{
				particles(row, col) = distribution(generator);
			}
		}
	}

	//Run the user supplied predict function across all particles
	void predict(function<Eigen::Vector<double, state_size>(Eigen::Vector<double, state_size>, Eigen::Vector<double, state_size>)> model, Eigen::Vector<double, state_size> u, Eigen::Vector<double, state_size> std_)
	{
		Eigen::Vector<double, state_size> temp = Eigen::Vector<double, state_size>::Zero();
		for (int row = 0; row < number; row++)
		{
			for (int col = 0; col < state_size; col++)
			{
				normal_distribution<double> distribution(0, std_(col));
				temp(col) = distribution(generator);
			}
			Eigen::Vector<double, state_size> noise = u + temp;
			particles.row(row) = model(noise, particles.row(row));
		}
	}

	//Take the average and variance of all the particles multiplied by their weights
	void estimate() 
	{
		Eigen::Vector<double, state_size> mean = Eigen::Vector<double, state_size>::Zero();
		for (int row = 0; row < number; row++)
		{
			mean += particles.row(row) * weights(row);
		}
		Eigen::Vector<double, state_size> var = Eigen::Vector<double, state_size>::Zero();
		for (int row = 0; row < number; row++)
		{
			Eigen::Vector<double, state_size> diff = particles.row(row) - mean.transpose();
			var += (weights(row) * diff.cwiseProduct(diff)) ;
		}
		var = number > 1 ? var / (number - 1) : var;
		state_mean = mean;
		state_var = var;
	}

	//Run update the weigths with estimates by the sensor with a user supplied update function
	void update(function<Eigen::Vector<double, measure_size>(Eigen::Vector<double, state_size>)> update_func, Eigen::Vector<double, measure_size> z, Eigen::Vector<double, measure_size> std_)
	{
		Eigen::Matrix<double, number, measure_size> transformed = Eigen::Matrix<double, number, measure_size>::Zero();
		for (int row = 0; row < number; row++)
		{
			transformed.row(row) = update_func(particles.row(row));
		}
		for (int col = 0; col < measure_size; col++)
		{
			for (int row = 0; row < number; row++)
			{
				weights(row) *= pdf(z(col), transformed.coeff(row, col), std_(col));
			}
		}
		weights /= weights.sum();
	}

	//a simple random sample of particles that are closest to the true state
	void resample() 
	{
		Eigen::Vector<double, number> cumulative_sum = Eigen::Vector<double, number>::Zero();
		Eigen::Matrix<double, number, state_size> resample_particles = Eigen::Matrix<double, number, state_size>::Zero();
		double sum = 0.0;
		int indexes[number];
		for (int row = 0; row < number; row++)
		{
			sum += weights[row];
			cumulative_sum(row) = sum;
		}
		cumulative_sum(number - 1) = 1.0;
		uniform_real_distribution<double> distrib(0.0, 1.0);
		for (int index = 0; index < number; index++)
		{
			double rand = distrib(generator);
			indexes[index] = (cumulative_sum.array() < rand).count();
		}
		for (int row = 0; row < number; row++)
		{
			resample_particles.row(row) = particles.row(indexes[row]);
		}
		particles = resample_particles;
		weights.fill(1.0 / ((double) number));
	}
	
	//calculates the Neffective value. This value determines if the user should resample the particles
	double neff() { 
		double sum_square = 0.0;
		for (int row = 0; row < number; row++)
		{
			sum_square += pow(weights(row), 2);
		}
		return 1.0 / sum_square;
	}

	//Getters and the probability density function
	inline double pdf(double x, double mew, double sigma) { return (1.0 / (sigma * sqrt(2.0 * pi))) * exp((-1.0 / 2.0) * pow((x - mew) / sigma, 2.0)); }
	inline Eigen::Matrix<double, number, state_size> get_particles() { return particles; }
	inline Eigen::RowVectorXd get_state_mean() { return state_mean; }
	inline Eigen::RowVectorXd get_state_var() { return state_var; }
	inline Eigen::Vector<double, number> get_weights() { return weights; }

	//All private varibles maintained by the class
private:
	Eigen::Matrix<double, number, state_size> particles = Eigen::Matrix<double, number, state_size>::Zero();
	Eigen::Vector<double, number> weights;
	Eigen::Vector<double, state_size> std_devs = Eigen::Vector<double, state_size>::Zero();
	Eigen::Vector<double, state_size> means = Eigen::Vector<double, state_size>::Zero();
	Eigen::Vector<double, state_size> state_mean = Eigen::Vector<double, state_size>::Zero();
	Eigen::Vector<double, state_size> state_var = Eigen::Vector<double, state_size>::Zero();
	std::default_random_engine generator;
	

};

//Test model function. Simply adds the noisy control input to the particle
Eigen::Vector<double, 2> model(Eigen::Vector<double, 2> noise, Eigen::Vector<double, 2> particles)
{
	Eigen::Vector<double, 2> ret{0.0, 0.0};
	ret(0) += noise(0) + particles(0);
	ret(1) += noise(1) + particles(1);
	return ret;
}

//Transitions the particle to measurement space
Eigen::VectorXd update(Eigen::Vector<double, 2> particles)
{
	Eigen::VectorXd resultant(3);
	resultant(0) = particles(0);
	resultant(1) = particles(1);
	resultant(2) = 2 * particles(0);
	return resultant;
}

//Main loop of the program
int main()
{
	Eigen::Vector<double, 2> tracker_state = { 0.0, 0.0 };
	ParticleFilter<2, 1000, 3> filter({0.0, 0.0}, { 1.0, 1.0 }); //declare and instantiate
	double measure_std = 1.0;
	std::default_random_engine gen;
	normal_distribution<double> distribution(0, measure_std);
	double limit = 5000.0/2.0;
	// initial estimate
	filter.estimate();
	cout << "Init State: " << filter.get_state_mean() << '\n';
	cout << "Init Std: " << filter.get_state_var() << '\n';
	Eigen::Vector<double, 2> temp = Eigen::Vector<double, 2>::Ones();
	for (int i = 0; i < 80; i++)
	{
		filter.predict(model, { 1.0, 1.0 }, { 0.5, 0.5 });
		tracker_state += temp;
		filter.update(update, { tracker_state(0) + distribution(gen), tracker_state(1) + distribution(gen),  tracker_state(0) + tracker_state(1) + distribution(gen)}, {measure_std, measure_std, measure_std});
		cout << "Tracker State at step " << i << ": " << tracker_state << '\n';
		cout << "State at step " << i << ": " << filter.get_state_mean() << '\n';
		cout << "Std at step " << i << ": " << filter.get_state_var() << '\n';
		cout << "Neff at step " << i << ": " << filter.neff() << '\n';
		if (filter.neff() < limit)
		{
			cout << "Neff < 25, reampling ... " << '\n';
			filter.resample();
		}
		filter.estimate();

	}
	
}
