package com.shumeau.hammerdrill.parallelization;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import no.uib.cipr.matrix.DenseMatrix;

import com.shumeau.hammerdrill.nn.NeuralNetworks;
import com.shumeau.hammerdrill.nn.Evaluations.Evaluation;


public class GradientParallelizer {
	
	public DataContainer parallelizeGradient(NeuralNetworks nn, int nb_Workers, int size_chunk, DenseMatrix samples, DenseMatrix targets, Evaluation evaluation) throws Exception{
		ExecutorService executor=Executors.newFixedThreadPool(nb_Workers);
		int total_nb_samples=samples.numColumns();
		DataContainer dc=new DataContainer(nn.getTotalNumberWeights(), total_nb_samples);
		int[] bounds_segments=indicesOfChunks(size_chunk, total_nb_samples);
		
		for(int i=0; i<bounds_segments.length-1;i++){
			NeuralNetworks nn_clone=nn.cloneFull();
			GradientWorker gw= new GradientWorker(dc,samples,targets,bounds_segments[i], bounds_segments[i+1], evaluation,nn_clone);
			executor.execute(gw);
		}
		executor.shutdown();
		try {
			executor.awaitTermination(1000, TimeUnit.SECONDS);
		} catch (InterruptedException e) {e.printStackTrace();}
		return dc;
	}
	
	private int[] indicesOfChunks(int size_chunks, int total_nb_samples){
		int nbChuncks=(total_nb_samples/size_chunks);
		if (total_nb_samples%size_chunks!=0)
			nbChuncks++;
		int[] indices=new int[nbChuncks+1];
		int counter=0;
		for (int i=0; i<=total_nb_samples; i+=size_chunks){
			indices[counter]=i;
			counter++;
		}
		if(counter<indices.length)
			indices[counter]=total_nb_samples;
		return indices;
	}

}
