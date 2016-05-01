float tanh_appro(float x)
{
	float op= (-0.67436811832e-5+(0.2468149110712040+(0.583691066395175e-1+0.3357335044280075e-1*x)*x)*x)/(0.2464845986383725+(0.609347197060491e-1+(0.1086202599228572+.2874707922475963e-1*x)*x)*x);
	return op;
}
float exp_appro(float x)
{
    float sum = 1;
 	int n=8; // Expand Taylor series by N 
    #pragma unroll 1
	for (int i = n - 1; i > 0; --i )
        sum = 1 + x * sum / i;
 
    return sum;
}
__attribute__((num_compute_units(1)))
__kernel void conv_kernel(  global float * restrict ip_img,  global float *restrict Weights, global float * restrict Wsoft, global float * restrict softmax_op, __constant float * restrict filt ,int Img_height, int Img_width , int sigm_no, int softmax_no)
{
	int idx = 0,tot_offset,f_pos,f_pos1;
	int FiltS = 3;
	float R_sum = 0.0, G_sum = 0.0, B_sum = 0.0;
	float rows[( ( (2) * 732) + 3) * 3];
	int iterations =  ( ( (FiltS-1) * Img_height) + FiltS) * 3;

	int count1 = Img_height * Img_width * 3;
	int iterations1 = count1;
	float sigmoids[100];
	float tempR[ 732 + 2 ];
	float tempG[ 732 + 2 ];
	float tempB[ 732 + 2 ];
	int cur_col =0,k=0;
//	printf(" iter = %d\t",iterations);
	for (count1 = 732*486*3;count1 > 0;count1=count1-3) 
	{	
		int i;
		#pragma unroll 1	
		for(i = 0; i < iterations; i= i+1) 
		{
			//printf("shift i = %d\n",i);
		    rows[i + 1] = rows[i]; //printf("rows[%d] = %f \t",i,rows[i]);
//		    rows[i + 2] = rows[i + 1]; //printf("rows[%d] = %f \t",i + 1,rows[i + 1]);
//		    rows[i + 3] = rows[i + 2]; //printf("rows[%d] = %f \n",i + 2,rows[i + 2]);
		}

		rows[0] = count1 >= 0 ? ip_img[iterations1 - count1] : 0;
		rows[1] = count1 >= 0 ? ip_img[iterations1 - count1 + 1] : 0;
		rows[2] = count1 >= 0 ? ip_img[iterations1 - count1 + 2] : 0;
		  
		//printf("ip rows[0] = %f \n",rows[0]);

		#pragma unroll	1		  		
		for( f_pos = 0; f_pos < 3; f_pos=f_pos+1 )
		{
			#pragma unroll 1
			for( f_pos1 = 0; f_pos1 < FiltS; f_pos1++ )
			{
				unsigned int idx = f_pos * FiltS + f_pos1;
				unsigned int cur_pixel = f_pos * Img_height + f_pos1;

				R_sum += rows[ cur_pixel ] * filt[ idx ]; 
				G_sum += rows[ cur_pixel + 1 ] * filt[ idx ];
				B_sum += rows[ cur_pixel + 2 ] * filt[ idx ];
			}
		}	

//		printf("Conv done %d Rsum = %f Gsum = %f Bsum = %f\t",count1,R_sum,G_sum,B_sum);
		int j;
		#pragma unroll 1
		for( j = 0 ; j < Img_height+1  ; j++ )
		{
			tempR[ j ] = tempR[ j + 1 ];
			tempG[ j ] = tempG[ j + 1 ];
			tempB[ j ] = tempB[ j + 1 ];	
		}

// Relu added
		int x = iterations1 - count1;

		if(R_sum < 0){R_sum = 0;}
		tempR[ Img_height + 1 ] = R_sum;
		R_sum = 0;

		if(G_sum < 0){G_sum = 0;}
		tempG[ Img_height + 1 ] = G_sum;
		G_sum = 0;

		if(B_sum < 0){B_sum = 0;}
		tempB[ Img_height + 1 ] = B_sum;
		B_sum = 0;

		if( x % (Img_height*3))
		{
			cur_col++;
		}

		//Max
		float max_R = 0; float max_G = 0; float max_B = 0;
		if( ( x > (Img_height*3 )) && ( x % 6  != 0 ) && ((cur_col % 2) != 0))
		{
			max_R = tempR[0];  max_G = tempG[0];  max_B = tempB[0];

			if( tempR[1] > tempR[0] )max_R = tempR[1];
			if( tempR[Img_height] > max_R) max_R = tempR[Img_height];
			if( tempR[Img_height + 1] > max_R) max_R = tempR[Img_height+1];

			if( tempG[1] > tempG[0] ) max_G = tempG[1];
			if( tempG[Img_height] > max_G) max_G = tempG[Img_height];
			if( tempG[Img_height + 1] > max_G) max_G = tempG[Img_height+1];

			if( tempB[1] > tempB[0] ) max_B = tempB[1];
			if( tempB[Img_height] > max_B) max_B = tempB[Img_height];
			if( tempB[Img_height + 1] > max_B) max_B = tempB[Img_height+1];

			//output[ k ]     = max_R;
			//output[ k + 1 ] = max_G;
			//output[ k + 2 ] = max_B;
			k+=3;
			
			
						
		
			
		}
			for(i=0;i<sigm_no;i++)
			{
				sigmoids[i] += Weights[k]     * max_R;
				sigmoids[i] += Weights[k+ 1] * max_G;
				sigmoids[i] += Weights[k + 2] * max_B;
			}
		
/*		output[ iterations1 - count1 ] = R_sum;
		R_sum = 0;
		
		output[ iterations1 - count1 + 1 ] = G_sum;
		G_sum = 0;

		output[ iterations1 - count1 + 2 ] = B_sum;
		B_sum = 0;

*/	//	printf("\nDebug k=%d",k);
	//	count1 -= 3 ;
	}
	
	int i;
	//#pragma unroll 1
	for (i=0;i<sigm_no;++i)
		sigmoids[i]=tanh_appro(sigmoids[i]);			
	// Runs only softmax_no times;
	//#pragma unroll 1
	for(i=0;i<softmax_no;++i)
	{	
	int j;
	float Z = 0;	
		for(j=0;j<sigm_no;++j)
			{
				Z += Wsoft[i*sigm_no + j]*sigmoids[j];
				
			}

		softmax_op[i]=exp_appro(Z);
	}		

	//one element only..
	
	float sum;	
	
	for(i=0;i<softmax_no;++i)
	{
		sum += softmax_op[i];		
	}
	
	for(i=0;i<softmax_no;++i)
	{
		softmax_op[i] = softmax_op[i] / sum;
		// Normalization		
	}	

}
