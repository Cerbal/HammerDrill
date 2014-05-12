package com.shumeau.hammerdrill.parallelization;

public class DataContainer {
	double[] gradient;
	double cost=0;
	int total_nb_samples;
	
	public DataContainer(int size_gradient, int total_nb_samples){
		gradient=new double[size_gradient];
		this.total_nb_samples=total_nb_samples;
	}
	
	public synchronized void updateData(double[] new_data,double cost, int nb_samples){
		double factor=1.0*nb_samples/total_nb_samples;
		for (int i=0; i<gradient.length; i++)
			gradient[i]+=new_data[i]*factor;
		this.cost+=cost*factor;
	}
	
	public double[] getGradient(){
		return gradient;
	}
	
	public double getCost(){
		return cost;
	}
}
