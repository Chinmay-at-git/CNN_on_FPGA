
#pragma OPENCL EXTENSION cl_altera_channels : enable

channel float pool2sigm;
channel float sigm2sigm;
channel float convo2pool_R;
channel float convo2pool_G;
channel float convo2pool_B;

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
__kernel void conv_kernel(global float* restrict ip_img, __constant float * restrict filt , int Img_width, int HFS)
{
	 int idx,tot_offset,f_pos,f_pos1;

		float R_sum , G_sum , B_sum;
	const int relu_const = 100;
	tot_offset = 3 * get_global_id(1) + ( get_global_id(0) * Img_width * 3 );
//	printf("\nImage Width=%d deviceids-(%d,%d)",Img_width,get_global_id(0),get_global_id(1));
	//printf("\ntot=%d",tot_offset);
	idx=0;
	R_sum=0.0 ; G_sum=0; B_sum=0;	
	for(f_pos = -HFS; f_pos <= HFS; f_pos++)
	{
		int cur_row = tot_offset + f_pos * (Img_width * 3);
		for(f_pos1 = -HFS; f_pos1 <= HFS; f_pos1++, idx += 1)
		{
			int cur_pix_offset = f_pos1 * 3;
			if((cur_pix_offset + cur_row+2) < 0)
			{
				continue;
			}	
			R_sum += ip_img[ cur_pix_offset + cur_row ] * filt[ idx ]; 
			G_sum += ip_img[ cur_pix_offset + cur_row + 1 ] * filt[ idx ];
			B_sum += ip_img[ cur_pix_offset + cur_row + 2 ] * filt[ idx  ];
		}
	}	
	if(tot_offset+2 < 0)
		return;
	if(R_sum < relu_const) R_sum = R_sum * 0.01; // Leaky RELU implementation
	if(G_sum < relu_const) G_sum = G_sum * 0.01; // Leaky RELU implementation
	if(B_sum < relu_const) B_sum = B_sum * 0.01; // Leaky RELU implementation
	write_channel_altera(convo2pool_R,R_sum );  // 
	write_channel_altera(convo2pool_G,G_sum );
	write_channel_altera(convo2pool_B,B_sum );
//	printf("\n%d %d RSUM=%f GSUM=%f BSUM=%f",get_global_id(0),get_global_id(1),R_sum,G_sum,B_sum);
}

__attribute__((num_compute_units(1)))
__kernel void sigm_kernel1(global float* restrict Weights,global float* restrict Wsoft,global float*restrict softmax_op, int poolH, int poolW,int sigm_no,int softmax_no)
{
//	printf("\nDeviceids{%d,%d}",get_global_id(0),get_global_id(1));
	 int R,point,N;
	  local float sigmoids[10];
	  float buffR[(486+1)],buffG[(486+1)],buffB[(486+1)]; // hardcode
	N=poolW*poolH*3;
	
	for(R=0;R<poolH;R++)
	{
	int C;
	for(C=0;C<poolW;C++)
		{
		 point = R*poolW*3 +C*3;
		 float imR,imG,imB;
		imR=read_channel_altera(convo2pool_R);
		imG=read_channel_altera(convo2pool_G);
		imB=read_channel_altera(convo2pool_B);
		int i;		
		//#pragma unroll 1 // shift register		
		for(i=0;i<486;i++)
		{
		buffR[i]=buffR[(i+1)];
		}
		buffR[i]=imR;
	
		//#pragma unroll 486 // shift register		
		for(i=0;i<486;i++)
		{
		buffG[i]=buffG[(i+1)];
		}
		buffG[i]=imG;
	
		//#pragma unroll 486 // shift register		
		for(i=0;i<486;i++)
		{
		buffB[i]=buffB[(i+1)];
		}
		buffB[i]=imB;
					
		//output[R*poolW*3+C*3+0]=imR;
		//output[R*poolW*3+C*3+1]=imG;
		//output[R*poolW*3+C*3+2]=imB;
		if(R%2!=0 && C%2!=0)
		{
			 float maxR,maxG,maxB;
			//check max from 4 pixels to pool
			maxR=buffR[1];
			if(buffR[0] > imR  ) imR =buffR[0];
			if(buffR[486] > buffR[1]) maxR= buffR[486];
			if(maxR < imR ) maxR=imR ; 

			maxG=buffB[1];
			if(buffG[0] > imG  ) imG =buffG[0];
			if(buffG[486] > buffG[1]) maxG= buffG[486];
			if(maxG < imG ) maxG=imG ; 
	
			maxB=buffB[1];
			if(buffB[0] > imB  ) imB =buffB[0];
			if(buffB[486] > buffB[1]) maxB= buffB[486];
			if(maxB < imB ) maxB=imB ;  
			int i;	
			for(i=0;i<sigm_no;++i)
			{
				
			sigmoids[i]+=Weights[i*N + point]*imR;
			sigmoids[i]+=Weights[i*N + point + 1]*imG;
			sigmoids[i]+=Weights[i*N + point + 2]*imB;
		
			}
		}
		}
	}

	int i;
	#pragma unroll 1
	for (i=0;i<sigm_no;++i)
		sigmoids[i]=tanh_appro(sigmoids[i]);			
	// Runs only softmax_no times;
	#pragma unroll 1
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

