#include <mpi.h>
#include <iostream>
#include <chrono>
#include "infoli.h"


void init_cellState(cellStateV &cs){
    for(auto &v : cs.Prev_V_Dend)   v = -60.0f;
    for(auto &v : cs.V_Dend)        v = -60.0f;
    for(auto &v : cs.Hcurrent_q)    v =  0.0337836f;
    for(auto &v : cs.Calcium_r)     v = 0.0112788f;
    for(auto &v : cs.Potassium_s)   v= 0.0049291f;
    for(auto &v : cs.I_CaH)         v= 0.5f;
    for(auto &v : cs.Ca2Plus)       v=3.7152f;
    for(auto &v : cs.Iapp)          v=0.0f;
    for(auto &v : cs.Ic)            v=0.0f;
    for(auto &v : cs.V_Soma)        v=-60.0f;
    for(auto &v : cs.Prev_V_Soma)   v=-60.0f;
    for(auto &v : cs.g_CaL)         v= 0.68f;
    for(auto &v : cs.Sodium_m)      v = 1.0127807f;
    for(auto &v : cs.Sodium_h)      v = 0.3596066f;
    for(auto &v : cs.Calcium_k)     v=0.7423159f;
    for(auto &v : cs.Calcium_l)     v=0.0321349f;
    for(auto &v : cs.Potassium_n)   v=0.2369847f;
    for(auto &v : cs.Potassium_p)   v=0.2369847f;
    for(auto &v : cs.Potassium_x_s) v=0.1f;
    for(auto &v : cs.V_Axon)        v=-60.0f;
    for(auto &v : cs.Prev_V_Axon)   v=-60.0f;
    for(auto &v : cs.Sodium_m_a)    v=0.003596066f;
    for(auto &v : cs.Sodium_h_a)    v=0.9f;
    for(auto &v : cs.Potassium_x_a) v=0.2369847;


}
int main(int argc, char** argv){
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const auto nsize = int(4);
    const auto step = int(10000);
    auto cs = cellStateV(nsize);
    init_cellState(cs);
    auto sc = SimConfig(nsize);
    auto time_start = std::chrono::system_clock::now();
    for(int i=0;i<step;i++){
        ComputeNetwork_neon(cs,sc);
	std::cout<<"Step: "<<i
		<<" vdend "<<  cs.Prev_V_Dend[0]
		//<<" vdend "<<  cs.V_Dend[0]
		//<<" iapp " <<  cs.Iapp[0]
		//<<" Ic "   <<  cs.Ic[0]
		<<" hcurr "<<  cs.Hcurrent_q[0]
		<<" calcr "<<  cs.Calcium_r[0]
		<<" potass "<< cs.Potassium_s[0]
		<<" icah "<<   cs.I_CaH[0]
		<<" Ca2Plus "<<cs.Ca2Plus[0]
		<<" vsoma "<<cs.Prev_V_Soma[0]
		//<<" vaxon "<<cs.Prev_V_Axon[0]
		<<"\n";

    }
    auto time_end = std::chrono::system_clock::now();
    auto end = std::chrono::duration_cast<std::chrono::microseconds>(time_end-time_start).count();
    std::cout<<"Out "<<cs.Prev_V_Axon[0]<<"\n";
    auto avg = float(float(end)/float(nsize));
    std::cout<<"time avg: "<< avg <<"\n";

    MPI_Finalize();

    return 0;
}
