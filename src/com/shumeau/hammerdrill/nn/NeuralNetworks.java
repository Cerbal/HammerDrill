package com.shumeau.hammerdrill.nn;

import java.io.IOException;
import java.util.ArrayList;

import com.jmatio.io.MatFileWriter;

import no.uib.cipr.matrix.DenseMatrix;

public class NeuralNetworks {
	/** The different layers*/
	ArrayList<Layer> layers=new ArrayList<Layer>();
	/** The total number of weight in this network (sum of the weight of 
	 *  each layers).
	 */
	int nbWeights; 
	
	/** in and out size of the samples */
	int in_size, out_size;
	
	/** The last cost that has been computed */
	double cost=0;
	
	public NeuralNetworks() {
	
	}
	/**
	 * Create a classic MLP with full connection (vanilla).
	 * @param numberOfUnits An array containing the number of units for each layer. You must also include
	 * the dimension of the input. E.g. {728,200,10} (input is dimension 728).
	 * @param activations An array of Activations.Activation (enumeration), one for each layer. E.g.
	 * {Activations.tanh,Activations.tanh}. 
	 */
	public NeuralNetworks(int[] numberOfUnits, Activations.Activation[] activations) {
		assert numberOfUnits.length>1 : "Length of numberOfUnits must be at least 2 (input and output)";
		assert numberOfUnits.length==activations.length+1 : "The number of layers and the size of 'activations' do not fit";
		createNN(numberOfUnits, activations);
	}
	
	/**
	 * Create a classic MLP with full connection (vanilla).
	 * @param numberOfUnits An array containing the number of units for each layer. You must also include
	 * the dimension of the input. E.g. {728,200,10} (input is dimension 728).
	 * @param activation a value Activations.Activation, the activation is supposed to be the same for every
	 * layer in this case.
	 */
	public NeuralNetworks(int[] numberOfUnits, Activations.Activation activation) {
		assert numberOfUnits.length>0 : "null number of units";
		Activations.Activation[] activations=new Activations.Activation[numberOfUnits.length-1];
		for (int i=0; i<numberOfUnits.length-1;i++)
			activations[i]=activation;
		createNN(numberOfUnits, activations);
	}
	
	/**
	 *  (private helper method) See constructor NeuralNetworks(int[] numberOfUnits, Activations.Activation[] activations)
	 *  for help.
	 */
	private void createNN(int[] numberOfUnits, Activations.Activation[] activations){
		nbWeights=0;
		for(int i=0; i<numberOfUnits.length-1; i++){
			LayerVanilla l=new LayerVanilla(numberOfUnits[i], numberOfUnits[i+1], activations[i]);
			layers.add(l);
			nbWeights+=l.getNumberWeights();
		}
		in_size=layers.get(0).getInSize();
		out_size=layers.get(layers.size()-1).getOutSize();
	}


	/**
	 * Apply the feedforward pass to the network. Notice the short code...
	 * @param samples DenseMatrix containing the samples. One sample per column.
	 * @return The output of the neural network. One sample per column.
	 */
	public DenseMatrix feedforward(DenseMatrix samples){
		DenseMatrix input=samples;
		for (Layer l : layers){
			input=l.feedforward(input);
		}
		return input;
	}
	
	/**
	 * Compute gradient toward all parameters
	 */
	public double[] feedforwardAndComputeGradient(DenseMatrix samples, DenseMatrix target, Evaluations.Evaluation evaluation){

		DenseMatrix output=feedforward(samples);

		DenseMatrix derivative_towards_output=new DenseMatrix(out_size, samples.numColumns());
		cost=Evaluations.applyEvaluation(output, target, derivative_towards_output, evaluation);
		DenseMatrix derivative_towards_input = null;
		/** backward pass */
		for(int i=layers.size()-1; i>=0; i--){
			if (i==0){
				layers.get(i).computeGradient(derivative_towards_output, samples);
			}
			else{
				derivative_towards_input=layers.get(i).computeGradient(derivative_towards_output, layers.get(i-1).getOutput());
				derivative_towards_output=derivative_towards_input;
			}
			layers.get(i).clear();
		}
		/** get the gradient of every layer and pack it into an array of double */
		double[] gradient=new double[nbWeights];
		int counter=0;
		for (Layer l : layers){
			double[] gradient_layer=l.packGradient();
			for (int i=0; i<gradient_layer.length; i++){
				gradient[counter]=gradient_layer[i];
				counter++;
			}
		}
		return gradient;
	}
	
	/**
	 * Assign coefficients to the Network.
	 */
	public void updateCoefficient(double[] theta){
		int counter=0;
		for (Layer l : layers){
			double[] gradient_layer=new double[l.getNumberWeights()];
			for (int i=0; i<l.getNumberWeights(); i++){
				gradient_layer[i]=theta[counter];
				counter++;
			}
			l.unpackCoefficients(gradient_layer);
		}
	}

	
	/**
	 * Get all the coefficients.
	 */
	public double[] getAllCoefficients(){
		/** get the coefficients of every layer and pack it into an array of double */
		double[] coeffs=new double[nbWeights];
		int counter=0;
		for (Layer l : layers){
			double[] coeffs_layer=l.packCoefficients();
			for (int i=0; i<coeffs_layer.length; i++){
				coeffs[counter]=coeffs_layer[i];
				counter++;
			}
		}
		return coeffs;
	}
	
	/**
	 * Get the last cost that has been computed.
	 * @return
	 */
	public double getCost(){
		return cost;
	}
	
	public int getTotalNumberWeights(){
		return nbWeights;
	}
	
	public double computeCost(DenseMatrix samples, DenseMatrix target, Evaluations.Evaluation evaluation){
		DenseMatrix output=feedforward(samples);
		DenseMatrix derivative_towards_output=new DenseMatrix(out_size, samples.numColumns());
		return Evaluations.applyEvaluation(output, target, derivative_towards_output, evaluation);
	}
	
	/**
	 * return a full clone with the coefficients for each layer
	 */
	public NeuralNetworks cloneFull(){
		ArrayList<Layer> copied_layers=new ArrayList<Layer>();
		for(Layer l : layers){
			copied_layers.add(l.cloneCoefs());
		}
		NeuralNetworks nn=new NeuralNetworks();
		nn.in_size=in_size;
		nn.layers=copied_layers;
		nn.nbWeights=nbWeights;
		nn.out_size=out_size;
		return nn;
	}

	/** return in size*/
	public int getInSize(){
		return in_size;
	}
	public int getOutSize(){
		return out_size;
	}
	
	/** Initialize network */
	public void initialize(){
		for(Layer l : layers){
			l.initialize();
		}
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static void saveOnMatlabFormat(NeuralNetworks nn, String path){
		ArrayList arrayList=new ArrayList();
		int counter=1;
		for(Layer l : nn.layers){
			l.toMatlabFormat(arrayList, counter);
			counter++;
		}
		 try {
			new MatFileWriter( path, arrayList );
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	

	
}
