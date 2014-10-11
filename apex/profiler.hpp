#ifndef PROFILER_HPP
#define PROFILER_HPP

//#include <boost/timer/timer.hpp>
#include <chrono>
#include <iostream>
#include <math.h>

using namespace std;
using namespace std::chrono;

namespace apex {

class profiler {
public:
        //boost::timer::cpu_timer t; // starts the timer when profiler is constructed!
	high_resolution_clock::time_point start;
	high_resolution_clock::time_point end;
	//boost::timer::cpu_times elapsed_time;
        void * action_address;
        string * timer_name;
	bool have_name;
        profiler(void * address) : action_address(address), have_name(false), start(high_resolution_clock::now()) {};
        profiler(string * name) : timer_name(name), have_name(true), start(high_resolution_clock::now()) {};
        ~profiler(void) {};
        void stop(void) { 
		//t.stop(); 
		end = high_resolution_clock::now();
		//elapsed_time = t.elapsed(); 
		//cout << t.format(boost::timer::default_places); 
	};
	double elapsed(void) { 
		duration<double> time_span = duration_cast<duration<double>>(end - start);
		return time_span.count(); 
	}
};

class profile {
private:
	int calls;
	double accumulated_time;
	double sum_squares;
	double minimum;
	double maximum;
public:
	profile(double elapsed) : calls(1), accumulated_time(elapsed), sum_squares(elapsed*elapsed), minimum(elapsed), maximum(elapsed) {};
	void increment(double elapsed) {
		accumulated_time += elapsed;
		sum_squares += (elapsed * elapsed);
		minimum = minimum > elapsed ? elapsed : minimum;
		maximum = maximum < elapsed ? elapsed : maximum;
		calls++;
	}
	void increment_resume(double elapsed) {
		accumulated_time += elapsed;
		// how to handle this?
		/*
		sum_squares += (elapsed * elapsed);
		minimum = minimum > elapsed ? elapsed : minimum;
		maximum = maximum < elapsed ? elapsed : maximum;
		*/
	}
	double get_calls() { return calls; }
	double get_mean() { return (accumulated_time / calls); }
	double get_minimum() { return (minimum); }
	double get_maximum() { return (maximum); }
	double get_variance() { 
		double mean = get_mean();
		return ((sum_squares / calls) - (mean * mean));
	}
	double get_stddev() { return sqrt(get_variance()); } 
};

}

#endif
