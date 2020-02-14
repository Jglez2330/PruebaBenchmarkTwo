#include <cmath>
#include <vector>	

	// Cell properties///////////////////////////////////////////////////////////////////////////////////////////////
#define DELTA 0.05
	//Conductance for neighbors' coupling
#define CONDUCTANCE 0.04
	// Capacitance
#define C_M 1
	// Somatic conductances (mS/cm2)
#define G_NA_S 150      // Na gate conductance (=90 in Schweighofer code, 70 in paper) 120 too little
#define G_KDR_S 9.0    // K delayed rectifier gate conductance (alternative value: 18)
#define G_K_S 5      // Voltage-dependent (fast) potassium
#define G_LS 0.016  // Leak conductance (0.015)
	// Dendritic conductances (mS/cm2)
#define G_K_CA 35       // Potassium gate conductance (35)
#define G_CAH 4.5     // High-threshold Ca gate conductance (4.5)
#define G_LD 0.016   // Dendrite leak conductance (0.015)
#define G_H 0.125    // H current gate conductance (1.5) (0.15 in SCHWEIGHOFER 2004)
	// Axon hillock conductances (mS/cm2)
#define G_NA_A 240      // Na gate conductance (according to literature: 100 to 200 times as big as somatic conductance)
#define G_NA_R 0      // Na (resurgent) gate conductance
#define G_K_A 20      // K voltage-dependent
#define G_LA 0.016  // Leak conductance
	// Cell morphology
#define P1 0.25        // Cell surface ratio soma/dendrite (0.2)
#define P2 0.15      // Cell surface ratio axon(hillock)/soma (0.1)
#define G_INT 0.13       // Cell internal conductance (0.13)
// Reversal potentials
#define V_NA 55       // Na reversal potential (55)
#define V_K -75       // K reversal potential
#define V_CA 120       // Ca reversal potential (120)
#define V_H -43       // H current reversal potential
#define V_L 10       // leak current

//typedef double mod_prec;
using  mod_prec = float;

struct cellStateV{
	std::vector<mod_prec> Prev_V_Soma;
	std::vector<mod_prec> Prev_V_Dend;
	std::vector<mod_prec> Prev_V_Axon;
	std::vector<mod_prec> V_Soma;
	std::vector<mod_prec> V_Dend;
	std::vector<mod_prec> V_Axon;
	/*Dend*/
	std::vector<mod_prec> Hcurrent_q;
	std::vector<mod_prec> Calcium_r;
	std::vector<mod_prec> Potassium_s;
	std::vector<mod_prec> I_CaH;
	std::vector<mod_prec> Ca2Plus;
	std::vector<mod_prec> Ic;
	std::vector<mod_prec> Iapp;
	/*Soma*/
	std::vector<mod_prec> g_CaL;
	std::vector<mod_prec> Sodium_m;
	std::vector<mod_prec> Sodium_h;
	std::vector<mod_prec> Calcium_k;
	std::vector<mod_prec> Calcium_l;
	std::vector<mod_prec> Potassium_n;
	std::vector<mod_prec> Potassium_p;
	std::vector<mod_prec> Potassium_x_s;
	/*Axon*/
	std::vector<mod_prec> Sodium_m_a;
	std::vector<mod_prec> Sodium_h_a;
	std::vector<mod_prec> Potassium_x_a;
	cellStateV(uint size): 
		Prev_V_Soma(size),
		Prev_V_Dend(size),
		Prev_V_Axon(size),
		V_Soma(size),
		V_Dend(size),
		V_Axon(size),
		/*Dend*/
		Hcurrent_q(size),
		Calcium_r(size),
		Potassium_s(size),
		I_CaH(size),
		Ca2Plus(size),
		Ic(size),
		Iapp(size),
		/*Soma*/
		g_CaL(size),
		Sodium_m(size),
		Sodium_h(size),
		Calcium_k(size),
		Calcium_l(size),
		Potassium_n(size),
		Potassium_p(size),
		Potassium_x_s(size),
		/*Axon*/
		Sodium_m_a(size),
		Sodium_h_a(size),
		Potassium_x_a(size)
		{};	
};

/*** FUNCTION PROTOTYPES ***/

struct SimConfig{
    int network_population=4;
    int local_population=network_population;
    int mpi_nodes;
    SimConfig(int net_pop, int mpi_nodes=1):
        network_population(net_pop),
        local_population(net_pop/mpi_nodes),
	mpi_nodes(mpi_nodes)
	{};
};


void ComputeNetwork_neon(cellStateV &, SimConfig &);

void CompDend_neon(cellStateV &s,uint N_Size);
void DendHCurr_neon(cellStateV & s,uint idx);
void DendCaCurr_neon(cellStateV &s, uint idx);
void DendKCurr_neon(cellStateV &s, uint idx);
void DendCal_neon(cellStateV &s, uint idx);
void DendCurrVolt_neon(cellStateV &s, uint idx);
void CompSoma_neon(cellStateV &s,uint N_Size);
void SomaCalcium_neon(cellStateV &st, uint idx);
void SomaSodium_neon(cellStateV &st, uint idx);
void SomaPotassium_neon(cellStateV &st, uint idx);
void SomaPotassiumX_neon(cellStateV &st, uint idx);
void SomaCurrVolt_neon(cellStateV &st, uint idx);
void CompAxon_neon(cellStateV &s,uint N_Size);
void AxonSodium_neon(cellStateV &st, uint idx);
void AxonPotassium_neon(cellStateV &st, uint idx);
void AxonCurrVolt_neon(cellStateV &st, uint idx);

void swapVec(cellStateV &s);

