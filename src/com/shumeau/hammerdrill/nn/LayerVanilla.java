package com.shumeau.hammerdrill.nn;

import java.util.ArrayList;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;

import com.jmatio.types.MLChar;
import com.jmatio.types.MLDouble;
import com.shumeau.hammerdrill.nn.Activations.Activation;

public class LayerVanilla extends Layer {
	Activation activation;
	DenseMatrix W,B;
	int in_size, out_size;
	
	/** output, Z A and the gradient toward Z, are stored here 
	 * */ 
	DenseMatrix Z;
	DenseMatrix A;
	DenseMatrix Z_derivative;
	
	/** Those variables receive the gradient toward W and B */
	DenseMatrix W_grad,B_grad;
	
	LayerVanilla(int in_size, int out_size, Activation activation){
		this.activation=activation;
		W=new DenseMatrix(out_size, in_size);
		B=new DenseMatrix(out_size,1);
		W_grad=new DenseMatrix(out_size, in_size);
		B_grad=new DenseMatrix(out_size, 1);
		
		this.in_size=in_size;
		this.out_size=out_size;
	}
	
	/**
	 * apply Z=WX+B
	 * A=activation(Z)
	 * Return a reference to A!!! for memory optimization. Do not modify the DenseMatrix which is 
	 * returned.
	 * @param X
	 */
	public DenseMatrix feedforward(DenseMatrix X){
		if (Z==null || Z.numColumns()!=X.numColumns() || Z.numRows()!=X.numRows()){
			Z=new DenseMatrix(out_size,X.numColumns());
			A=new DenseMatrix(out_size,X.numColumns());
			Z_derivative=new DenseMatrix(out_size,X.numColumns());
		}

		/** Z=WxX */
		W.mult(X, Z);


		/** Z=Z+b */
		int counter=0;

		for(MatrixEntry me : Z){
			me.set(me.get()+B.get(counter, 0));
			counter++;
			counter=counter%Z.numRows();
		}

		/** activation */
		long tic=0;
		if (Debug.verbose_computation_time)
			tic=System.currentTimeMillis();
		
		Activations.applyActivation(Z, A, Z_derivative, activation);	
		
		if (Debug.verbose_computation_time){
			long toc=System.currentTimeMillis();
			Debug.addActivationTime((int) (toc-tic));
		}
		
		
		/** Z is useless now */
		Z=null;
		return A;
	}
	
	/** Compute the gradient.
	 * Following standard convention of writing, if this is layer l,
	 * then it receives r_l, which has been computed above, and a_l-1, which has
	 * been computed below.
	 * It outputs r_l-1 and update W_grad and B_grad
	 */
	DenseMatrix computeGradient(DenseMatrix R_l, DenseMatrix A_l_1){
		/** multiply the Z derivatives by R_l to obtain RZ 
		 * The name of variables are simply kept for clarity*/
		DenseMatrix RZ_l=Z_derivative;
		Z_derivative=null;
		for (int i=0; i<RZ_l.getData().length; i++){
			RZ_l.getData()[i]*=R_l.getData()[i];
		}
		/**
		for (MatrixEntry me : RZ_l){
			me.set(me.get()*R_l.get(me.row(), me.column()));
		}
		*/
		/** compute the gardient toward W */
		RZ_l.transBmult(1.0/A_l_1.numColumns(),A_l_1, W_grad);
		
		/** compute the gradient toward B*/
		for(int i=0;i<RZ_l.numRows(); i++){
			double sum=0;
			for (int j=0;j<RZ_l.numColumns(); j++){
				sum+=RZ_l.getData()[i+j*RZ_l.numRows()];
			}
			B_grad.set(i,0,sum/A_l_1.numColumns());
		}

		/** compute the gradient toward the input */
		DenseMatrix R_l_1=new DenseMatrix(A_l_1.numRows(), A_l_1.numColumns());
		
		W.transAmult(RZ_l, R_l_1);
		RZ_l=null;
		return R_l_1;
		
	}
	
	/**
	 * Pack the coefficients of the gradient (W_grad and B_grad) into a single
	 * double array. Order in the following : [W_grad(column0); ..; W_grad(columnN),B_grad]
	 * @return
	 */
	double[] packGradient(){
		return pack(W_grad, B_grad);
	}
	
	private double[] pack(DenseMatrix DM1, DenseMatrix DM2){
		assert(DM1!=null) : "gradient seems not to have been computed yet";
		double[] theta=new double[(in_size+1)*out_size];
		int counter=0;
		for (int i=0; i<DM1.numColumns(); i++){
			for (int j=0; j<DM1.numRows(); j++){
				theta[counter]=DM1.get(j, i);
				counter++;
			}
		}
		for (int j=0; j<DM2.numRows(); j++){
			theta[counter]=DM2.get(j, 0);
			counter++;
		}
		return theta;
	}

	double[] packCoefficients() {
		return pack(W, B);
	}

	void unpackCoefficients(double[] theta) {
		assert ((in_size+1)*out_size==theta.length) : "Problem during unpacking : number of coefficient do not fit";
		int counter=0;
		for (int i=0; i<W.numColumns(); i++){
			for (int j=0; j<W.numRows(); j++){
				W.set(j, i, theta[counter]);
				counter++;
			}
		}
		for (int j=0; j<B.numRows(); j++){
			B.set(j, 0, theta[counter]);
			counter++;
		}
	}

	@Override
	int getNumberWeights() {
		return (in_size+1)*out_size;
	}

	@Override
	DenseMatrix getOutput() {
		return A;
	}

	@Override
	int getInSize() {
		return in_size;
	}

	@Override
	int getOutSize() {
		return out_size;
	}

	@Override
	Layer cloneStructure() {
		return new LayerVanilla(in_size, out_size, activation);
	}
	
	@Override
	Layer cloneCoefs() {
		LayerVanilla l=new LayerVanilla(in_size, out_size, activation);
		l.B=new DenseMatrix(B);
		l.W=new DenseMatrix(W);
		return l;
		
	}

	@Override
	void initialize() {
		/** Initialize W with Gaussian distribution (so far) */
		W=Utils.createRandomMatrix(out_size, in_size);
	}

	@Override
	void clear() {
		this.A=null;
		this.Z=null;
		this.Z_derivative=null;

	}

	@SuppressWarnings("unchecked")
	@Override
	void toMatlabFormat(@SuppressWarnings("rawtypes") ArrayList arrayList, int id_layer) {
		MLDouble mW = new MLDouble( "W_layer"+String.valueOf(id_layer), W.getData(), W.numRows() );
		MLDouble mB = new MLDouble( "B_layer"+String.valueOf(id_layer), B.getData(), B.numRows() );
		MLChar mact=new MLChar("activation_layer"+String.valueOf(id_layer), Activations.toString(activation));
		arrayList.add(mW);
		arrayList.add(mB);
		arrayList.add(mact);
	}
	
	
}
