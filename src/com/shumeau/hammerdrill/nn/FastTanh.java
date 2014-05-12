package com.shumeau.hammerdrill.nn;

/**
 * This class contains a faster implementation of tanh. Indeed the one given in the JVM
 * appears to be quite slow, and the computation of the activation function has become one of
 * the major time waste. Thus, this class provides a linear and quadratic approximation of tanh.
 * In both casesthe absolute error is always below 2e-7 and the relative one is always below
 * 6e-6.
 * @author pcSamuel
 *
 */
public class FastTanh {
	/** step of the approximation */
	final int N_quad=1024;
	final int N_lin=5000;
	/** maximum of the approximation */
	final int max_approx_quad=9;
	final int max_approx_lin=8;
	/** flag that tells if it is prepared for tanh*/
	boolean prepared_for_tanh=false;
	boolean prepared_for_tanh2=false;
	/** array containing the X values */
	double[] xs;
	/** coefficient for the quadratic approximation of tanh */
	double[] a,b,c;
	/** coefficient for the linear approximation of tanh */
	double[] d,e;
	
	/** get the last gradient computed */
	double last_gradient;
	
	/** prepare the values necessary to the computation of the tan approximation */
	void prepareForTanh(){
		d=new double[N_lin-1];
		e=new double[N_lin-1];
		xs=new double[N_lin];
		/** note, tanh(0)=0,but otherwise we would have to initialize xs[0] */
		double x1,x2;
		for (int i=1; i<N_lin; i++){
			xs[i]=1.0*i*max_approx_lin/(N_lin-1);
		    x1=xs[i-1];
		    x2=xs[i];
		    d[i-1]=(Math.tanh(x2)-Math.tanh(x1))/(x2-x1);
		    e[i-1]=Math.tanh(x1)-d[i-1]*x1;
		}
		prepared_for_tanh=true;
	}
	
	/** linear approximation of tanh*/
	double tanh(double x){
		if (!prepared_for_tanh)
			prepareForTanh();
		double x_pos;
		if (x<0) x_pos=-x; else x_pos=x;
		if (x_pos>=max_approx_lin){
			last_gradient=0;
			if (x<0) return -1; 
			return 1;
		}
		int indice=(int) (x_pos*(N_lin-1)/max_approx_lin);
		double approx=d[indice]*x_pos+e[indice];
		last_gradient=d[indice];
		if (x<0) return -approx; 
		return approx;
	}
	
	/** prepare the values necessary to the computation of the tan approximation */
	void prepareForTanh2(){
		a=new double[N_quad-1];
		b=new double[N_quad-1];
		c=new double[N_quad-1];
		xs=new double[N_quad];
		/** note, tanh(0)=0,but otherwise we would have to initialize xs[0] */
		double x1,x2,y1,z1,z2,z3,z4,z5,z6;
		for (int i=1; i<N_quad; i++){
			xs[i]=1.0*i*max_approx_quad/(N_quad-1);
		    x1=xs[i-1];
		    x2=xs[i];
		    y1=(xs[i-1]+xs[i])/2;
		    z1=(x1-y1)*(x1+y1);
		    z2=x1-y1;
		    z3=Math.tanh(x1)-Math.tanh(y1);
		    z4=(x2-y1)*(x2+y1);
		    z5=x2-y1;
		    z6=Math.tanh(x2)-Math.tanh(y1);
		    a[i-1]=(z3*z5-z2*z6)/(z1*z5-z2*z4);
		    b[i-1]=(z3-a[i-1]*z1)/z2;
		    c[i-1]=tanh(x1)-b[i-1]*x1-a[i-1]*x1*x1;
		}
		c[0]=0;
		b[0]=1;
		a[0]=(Math.tanh(xs[1])-xs[1])/(xs[1]*xs[1]);
		prepared_for_tanh2=true;
	}
	
	/** quadratic approximation of tanh. */
	double tanh2(double x){
		if (!prepared_for_tanh2)
			prepareForTanh2();
		double x_pos;
		if (x<0) x_pos=-x; else x_pos=x;
		if (x_pos>=max_approx_quad){
			if (x<0) return -1; 
			return 1;
		}
		int indice=(int) (x_pos*(N_quad-1)/max_approx_quad);
		double approx=a[indice]*x_pos*x_pos+b[indice]*x_pos+c[indice];
		if (x<0) return -approx; 
		return approx;
	}
}
