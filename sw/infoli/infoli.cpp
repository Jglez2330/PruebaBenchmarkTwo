#include <thread>
#include "infoli.h"
#include "../neon_math/neon_mathfun.h"

inline auto oneVec(float a){return vdupq_n_f32(a);}
inline mod_prec min(mod_prec a, mod_prec b){return (a<b)?a:b;}

/*
 * f(a,b)=a/b
 */
auto divVec(float32x4_t a, float32x4_t b) {
	auto rec = vrecpeq_f32(b);
	rec = vmulq_f32(vrecpsq_f32(b, rec),rec);
	return vmulq_f32(a,rec);
}

void swapVec(cellStateV &s){
    s.Prev_V_Dend.swap(s.V_Dend);
    s.Prev_V_Soma.swap(s.V_Soma);
    s.Prev_V_Axon.swap(s.V_Axon);
}

void CompDend_neon(cellStateV &s, uint N_Size) {

    for(uint idx=0;idx<N_Size;idx+=4){
        DendHCurr_neon(s,idx);
        DendCaCurr_neon(s,idx);
        DendKCurr_neon(s,idx);
        DendCal_neon(s,idx);
        DendCurrVolt_neon(s,idx);
    }
}



void
DendHCurr_neon(cellStateV &s, uint idx) {
    using v4sf = float32x4_t;
    auto const src = s.Hcurrent_q.data()+idx;
    auto const prevV_dend = vld1q_f32(s.Prev_V_Dend.data()+idx);
    auto const exp_qinf = exp_ps(vaddq_f32(vmulq_f32(prevV_dend,oneVec(0.25f)),oneVec(20.0f)));
    auto const q_inf = divVec(oneVec(1.0f),(vaddq_f32(exp_qinf,oneVec(1.0f))));
    auto const exp0_tau = exp_ps(vaddq_f32(vmulq_f32(prevV_dend,oneVec(-0.086f)),oneVec(-14.6f)));
    auto const exp1_tau = exp_ps(vaddq_f32(vmulq_f32(prevV_dend,oneVec(0.070f)),oneVec(-1.87f)));
    auto const inv_tau_q = vaddq_f32(exp0_tau,exp1_tau);
    auto const prevHcurrent = vld1q_f32(src);
    auto const dq_dt = vmulq_f32(vaddq_f32(q_inf,vnegq_f32(prevHcurrent)),inv_tau_q);
    auto const delta = vdupq_n_f32(DELTA);
    auto const q_local = vaddq_f32(vmulq_f32(dq_dt,delta),prevHcurrent);
    vst1q_f32(src,q_local);
}




void DendCaCurr_neon(cellStateV &s, uint idx){
    //opt for division by constant
    auto const div_five = 0.2f;
    auto const div_thirteen = -1.0f/13.9f;
    auto const delta_five = DELTA*0.2f;

    //Get inputs
    auto const src = s.Calcium_r.data()+idx;
    auto const prevV_dend = vld1q_f32(s.Prev_V_Dend.data()+idx);
    auto const prevCalcium_r = vld1q_f32(src);

    auto const exp_alpha = exp_ps(vmulq_f32(vaddq_f32(prevV_dend,oneVec(-5.0f)),oneVec(div_thirteen)));
    auto const alpha_r = divVec(oneVec(1.7f),vaddq_f32(oneVec(1.0f),exp_alpha));
    auto const vplus_eight = vaddq_f32(prevV_dend,oneVec(8.5f)); 
    auto const exp_beta = exp_ps(vmulq_f32(vplus_eight,oneVec(div_five)));
    auto const beta_r = divVec(vmulq_f32(oneVec(0.02f),vplus_eight),vaddq_f32(exp_beta,oneVec(-1.0f)));
    auto const alpha_p_beta = vmulq_f32(oneVec(delta_five),vaddq_f32(alpha_r,beta_r));
    auto const r_local = vaddq_f32(vmulq_f32(vsubq_f32(oneVec(1.0f),alpha_p_beta),prevCalcium_r),vmulq_f32(oneVec(delta_five),alpha_r));
    vst1q_f32(src,r_local);
}



void DendKCurr_neon(cellStateV &s, uint idx){
    auto const beta_s = 0.015f;
    auto const beta_s_delta_one = beta_s*DELTA-1.0f;

    //Get inputs
    auto const src = s.Potassium_s.data()+idx;
    auto const prevPotassium_s = vld1q_f32(src);
    auto const prevCa2Plus = vld1q_f32(s.Ca2Plus.data()+idx);

    // Update dendritic Ca-dependent K current component
    auto const alpha_s = vminq_f32(vmulq_f32(oneVec(0.00002f),prevCa2Plus),oneVec(0.01f));
    auto const delta_alpha = vmulq_f32(oneVec(DELTA),alpha_s);
    auto const s_local = vsubq_f32(delta_alpha,vmulq_f32(vaddq_f32(delta_alpha,oneVec(beta_s_delta_one)),prevPotassium_s));
    vst1q_f32(src,s_local); 
}



void DendCal_neon(cellStateV &s, uint idx){
    auto const one_delta_sevenfive = 1.0f-DELTA*0.075f;
    auto const minus_three_delta = -3.0f*DELTA;

    //Get inputs
    auto const src = s.Ca2Plus.data()+idx;
    auto const prevCa2Plus = vld1q_f32(src);
    auto const prevI_CaH = vld1q_f32(s.I_CaH.data()+idx);
    
    // update Calcium concentration
    auto const Ca2Plus_local = 
	    vaddq_f32(vmulq_f32(oneVec(minus_three_delta),prevI_CaH),
		    vmulq_f32(oneVec(one_delta_sevenfive),prevCa2Plus));
    vst1q_f32(src,Ca2Plus_local);
}



void DendCurrVolt_neon(cellStateV &st, uint idx){
    auto constexpr inv_C_M = DELTA/C_M;
    auto constexpr ISD_CONST  = (G_INT*DELTA/(1.0f - P1))/C_M;
    auto constexpr ICAH_CONST = G_CAH*DELTA/ C_M;
    auto constexpr IKCA_CONST = G_K_CA*DELTA/ C_M;
    auto constexpr ILD_CONST  = G_LD*DELTA/ C_M;
    auto constexpr IH_CONST = G_H*DELTA/ C_M;

    auto const src = st.Prev_V_Dend.data()+idx;
    auto const src_ica = st.I_CaH.data()+idx;
    
    //Get inputs
    auto const I_c = vld1q_f32(st.Ic.data()+idx);
    auto const I_app = vld1q_f32(st.Iapp.data()+idx);
    auto const prevV_dend = vld1q_f32(src);
    auto const prevV_soma = vld1q_f32(st.Prev_V_Soma.data()+idx);
    auto const q = vld1q_f32(st.Hcurrent_q.data()+idx);
    auto const r = vld1q_f32(st.Calcium_r.data()+idx);
    auto const s = vld1q_f32(st.Potassium_s.data()+idx);

    // DENDRITIC CURRENTS
    // Soma-dendrite interaction current I_sd
    auto const I_sd = vmulq_f32(oneVec(ISD_CONST),vsubq_f32(prevV_dend,prevV_soma));
    // Inward high-threshold Ca current I_CaH
    auto const I_CaH = vmulq_f32(oneVec(ICAH_CONST),vmulq_f32(vmulq_f32(r,r),vsubq_f32(prevV_dend,oneVec(V_CA))));
    // Outward Ca-dependent K current I_K_Ca
    auto const I_K_Ca = vmulq_f32(oneVec(IKCA_CONST), vmulq_f32(s, vsubq_f32(prevV_dend,oneVec(V_K))));
    // Leakage current I_ld
    auto const I_ld = vmulq_f32(oneVec(ILD_CONST),vsubq_f32(prevV_dend,oneVec(V_L)));
    // Inward anomalous rectifier I_h
    auto const I_h = vmulq_f32(oneVec(IH_CONST),vmulq_f32(q,vsubq_f32(prevV_dend,oneVec(V_H))));

    auto const sumatory = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(I_CaH,I_sd),I_ld),I_K_Ca),I_c),I_h);
    auto const delta_dVd_dt = vsubq_f32(vmulq_f32(I_app,oneVec(inv_C_M)), sumatory);
    
    auto const Vdend = vaddq_f32(delta_dVd_dt,prevV_dend);
    vst1q_f32(st.V_Dend.data()+idx,Vdend);
    vst1q_f32(st.I_CaH.data()+idx,I_CaH);

}



void CompSoma_neon(cellStateV &s,uint N_Size){

    for(uint idx=0;idx<N_Size;idx+=4){
        SomaCalcium_neon(s,idx);
        SomaSodium_neon(s,idx);
        SomaPotassium_neon(s,idx);
        SomaPotassiumX_neon(s,idx);
        SomaCurrVolt_neon(s,idx);
    }
}




void SomaCalcium_neon(cellStateV &st, uint idx){
    
    //opt for division with constant
    auto const four = 1.0f/4.2;
    auto const eight = 1.0f/8.5; //no change in area by synthesizer on V7
    auto const thirty = 1.0/30.0;  //no change in area by synthesizer on V7
    auto const seven = 1.0/7.3; //no change in area by synthesizer on V7

    //Get inputs
    auto const prevV_soma = vld1q_f32(st.Prev_V_Soma.data()+idx);
    auto const prevCalcium_k = vld1q_f32(st.Calcium_k.data()+idx);
    auto const prevCalcium_l = vld1q_f32(st.Calcium_l.data()+idx);

    auto const exp_k_inf = exp_ps(vmulq_f32(vmulq_f32(vaddq_f32(prevV_soma,oneVec(61.0f)),oneVec(four)),oneVec(-1.0f)));
    auto const k_inf = divVec(oneVec(1.0f),vaddq_f32(exp_k_inf,oneVec(1.0f)));

    auto const exp_l_inf = exp_ps(vmulq_f32(vaddq_f32(prevV_soma,oneVec(85.5f)),oneVec(eight)));
    auto const l_inf = divVec(oneVec(1.0f),vaddq_f32(exp_l_inf,oneVec(1.0f)));

    auto const tau_l_num = vmulq_f32(exp_ps(vmulq_f32(vaddq_f32(prevV_soma,oneVec(160.0f)),oneVec(thirty))),oneVec(20.0f));
    auto const tau_l_den = vaddq_f32(exp_ps(vmulq_f32(vaddq_f32(prevV_soma,oneVec(84.0f)),oneVec(seven))),oneVec(1.0f));
    auto const tau_l = vaddq_f32(divVec(tau_l_num,tau_l_den),oneVec(35.0f));

    auto const dk_dt = vsubq_f32(k_inf,prevCalcium_k);
    auto const dl_dt = divVec(vsubq_f32(l_inf,prevCalcium_l),tau_l);
    auto const k_local = vaddq_f32(vmulq_f32(oneVec(DELTA),dk_dt),prevCalcium_k);
    auto const l_local = vaddq_f32(vmulq_f32(oneVec(DELTA),dl_dt),prevCalcium_l);
    //Put result
    vst1q_f32(st.Calcium_k.data()+idx,k_local);
    vst1q_f32(st.Calcium_l.data()+idx,l_local);

}



void SomaSodium_neon(cellStateV &st, uint idx){
    //opt for division by constant
    auto const five_five = 1.0f/5.5f; //no change in area by synthesizer on V7
    auto const five_eight = -1.0f/5.8f; //no change in area by synthesizer on V7
    auto const thirty_three = 1.0f/33.0f;  //no change in area by synthesizer on V7  

    //Get inputs
    auto const prevV_soma = vld1q_f32(st.Prev_V_Soma.data()+idx);
    auto const prevSodium_h = vld1q_f32(st.Sodium_h.data()+idx);

    // RAT THALAMOCORTICAL SODIUM:

    auto const exp_m_inf = exp_ps(vmulq_f32(vsubq_f32(oneVec(-30.0f),prevV_soma),oneVec(five_five)));
    auto const m_inf = divVec(oneVec(1.0f),vaddq_f32(exp_m_inf,oneVec(1.0f)));

    auto const exp_h_inf = exp_ps(vmulq_f32(vsubq_f32(oneVec(-70.0f),prevV_soma),oneVec(five_eight)));
    auto const h_inf = divVec(oneVec(1.0f),vaddq_f32(exp_h_inf,oneVec(1.0f)));

    auto const exp_tau_h = exp_ps(vmulq_f32(vsubq_f32(oneVec(-40.0f),prevV_soma),oneVec(thirty_three)));
    auto const tau_h = vmulq_f32(exp_tau_h,oneVec(3.0f));

    auto const dh_dt = divVec(vsubq_f32(h_inf,prevSodium_h),tau_h);
    auto const m_local = m_inf;
    auto const h_local = vaddq_f32(vmulq_f32(dh_dt,oneVec(DELTA)),prevSodium_h);
    //Put result
    vst1q_f32(st.Sodium_m.data()+idx,m_local);
    vst1q_f32(st.Sodium_h.data()+idx,h_local);

}



void SomaPotassium_neon(cellStateV &st, uint idx){

    //opt for division by constant
    auto const ten = 0.1f; //precision error if 1/10 is used
    auto const twelve = 1.0f/12.0f; //no change in area by synthesizer on V7
    auto const nine_hundred = 1.0f/900.0f;  //no change in area by synthesizer on V7

    //Get inputs
    auto const prevV_soma = vld1q_f32(st.Prev_V_Soma.data()+idx);
    auto const prevPotassium_n = vld1q_f32(st.Potassium_n.data()+idx);
    auto const prevPotassium_p = vld1q_f32(st.Potassium_p.data()+idx);

    // NEOCORTICAL
    auto const exp_n_inf = exp_ps(vmulq_f32(vaddq_f32(vnegq_f32(prevV_soma),oneVec(-3.0f)),oneVec(ten)));
    auto const n_inf = divVec(oneVec(1.0f),vaddq_f32(exp_n_inf,oneVec(1.0f)));
    auto const exp_p_inf = exp_ps(vmulq_f32(vaddq_f32(prevV_soma,oneVec(51.0f)),oneVec(twelve)));
    auto const p_inf = divVec(oneVec(1.0f),vaddq_f32(exp_p_inf,oneVec(1.0f)));
    auto const exp_tau_n = exp_ps(vmulq_f32(vaddq_f32(prevV_soma,oneVec(50.0f)),oneVec(nine_hundred)));
    auto const tau_n = vaddq_f32(vmulq_f32(exp_tau_n,oneVec(47.0f)),oneVec(5.0f));

    auto const dn_dt = divVec(vaddq_f32(vnegq_f32(prevPotassium_n),n_inf),tau_n);
    auto const dp_dt = divVec(vaddq_f32(vnegq_f32(prevPotassium_p),p_inf),tau_n);

    auto const n_local = vaddq_f32(vmulq_f32(dn_dt,oneVec(DELTA)),prevPotassium_n);
    auto const p_local = vaddq_f32(vmulq_f32(dp_dt,oneVec(DELTA)),prevPotassium_p);

    //Put result
    vst1q_f32(st.Potassium_n.data()+idx,n_local);
    vst1q_f32(st.Potassium_p.data()+idx,p_local);
}



void SomaPotassiumX_neon(cellStateV &st, uint idx){

    //opt for division by constant
    auto const ten = 0.1f; //no change in area by synthesizer on V7

    //Get inputs
    auto const prevV_soma = vld1q_f32(st.Prev_V_Soma.data()+idx);
    auto const prevPotassium_x_s = vld1q_f32(st.Potassium_x_s.data()+idx);

    // Voltage-dependent (fast) potassium
    auto const exp_alpha_x_s = exp_ps(vmulq_f32(vnegq_f32(vaddq_f32(prevV_soma,oneVec(25.0f))),oneVec(ten)));
    auto const den_alpha_x_s = vaddq_f32(vnegq_f32(exp_alpha_x_s),oneVec(1.0f));
    auto const alpha_x_s = divVec(vmulq_f32(vaddq_f32(prevV_soma,oneVec(25.0f)),oneVec(0.13f)),den_alpha_x_s);

    auto const exp_beta_x_s = exp_ps(vmulq_f32(vaddq_f32(prevV_soma,oneVec(35.0f)),oneVec(-0.0125f)));
    auto const beta_x_s  = vmulq_f32(exp_beta_x_s,oneVec(1.69f));

    auto const sum_x_s_local = vaddq_f32(vmulq_f32(vnegq_f32(vaddq_f32(alpha_x_s,beta_x_s)),oneVec(DELTA)),oneVec(1.0f));
    auto const x_s_local = vaddq_f32(vmulq_f32(oneVec(DELTA),alpha_x_s),vmulq_f32(sum_x_s_local,prevPotassium_x_s));

    //Put result
    vst1q_f32(st.Potassium_x_s.data()+idx,x_s_local);

}



void SomaCurrVolt_neon(cellStateV &st, uint idx){

    auto constexpr inv_CM = DELTA/C_M;
    auto constexpr div_GINT_P1 = (G_INT / P1);
    auto constexpr div_GINT_P2 = (G_INT / (1.0f - P2));

    //Get inputs
    auto const g_CaL = vld1q_f32(st.g_CaL.data()+idx);
    auto const prevV_dend = vld1q_f32(st.Prev_V_Dend.data()+idx);
    auto const prevV_soma = vld1q_f32(st.Prev_V_Soma.data()+idx);
    auto const prevV_axon = vld1q_f32(st.Prev_V_Axon.data()+idx);
    auto const k = vld1q_f32(st.Calcium_k.data()+idx);
    auto const l = vld1q_f32(st.Calcium_l.data()+idx);
    auto const m = vld1q_f32(st.Sodium_m.data()+idx);
    auto const h = vld1q_f32(st.Sodium_h.data()+idx);
    auto const n = vld1q_f32(st.Potassium_n.data()+idx);
    auto const x_s = vld1q_f32(st.Potassium_x_s.data()+idx);

    // Dendrite-soma interaction current I_ds
    auto const I_ds  = vmulq_f32(vaddq_f32(vnegq_f32(prevV_dend),prevV_soma),oneVec(div_GINT_P1));
    // Inward low-threshold Ca current I_CaL
    auto const k_exp_3 = vmulq_f32(vmulq_f32(k,k),k);
    auto const I_CaL = vmulq_f32(vmulq_f32(vmulq_f32(vaddq_f32(prevV_soma,oneVec(-V_CA)),l),k_exp_3),g_CaL);
    // Inward Na current I_Na_s
    auto const m_exp_3 = vmulq_f32(vmulq_f32(m,m),m);
    auto const I_Na_s  = vmulq_f32(vmulq_f32(vmulq_f32(vaddq_f32(prevV_soma,oneVec(-V_NA)),h),m_exp_3),oneVec(G_NA_S));
    // Leakage current I_ls
    auto const I_ls  = vmulq_f32(oneVec(G_LS),vaddq_f32(prevV_soma,oneVec(-V_L)));
    // Outward delayed potassium current I_Kdr
    auto const n_exp_4 = vmulq_f32(vmulq_f32(vmulq_f32(n,n),n),n);
    auto const I_Kdr_s = vmulq_f32(vmulq_f32(vaddq_f32(prevV_soma,oneVec(-V_K)),n_exp_4),oneVec(G_KDR_S)); // SCHWEIGHOFER
    // I_K_s
    auto const x_s_exp_4 = vmulq_f32(vmulq_f32(vmulq_f32(x_s,x_s),x_s),x_s);
    auto const I_K_s = vmulq_f32(vmulq_f32(vaddq_f32(prevV_soma,oneVec(-V_K)),x_s_exp_4),oneVec(G_K_S));
    // Axon-soma interaction current I_as
    auto const I_as = vmulq_f32(vaddq_f32(vnegq_f32(prevV_axon),prevV_soma),oneVec(div_GINT_P2));

    auto const sumatory = vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(vaddq_f32(I_CaL,I_ds),I_as),I_Na_s),I_ls),I_Kdr_s),I_K_s);
    auto const dVs_dt = vmulq_f32(vnegq_f32(sumatory),oneVec(inv_CM));

    //Put result
    auto const newVSoma = vaddq_f32(vmulq_f32(oneVec(DELTA),dVs_dt),prevV_soma);
    vst1q_f32(st.V_Soma.data()+idx,newVSoma);
}



void CompAxon_neon(cellStateV &s,uint N_Size){

    for(uint idx=0;idx<N_Size;idx+=4){
        AxonSodium_neon(s,idx);
        AxonPotassium_neon(s,idx);
        AxonCurrVolt_neon(s,idx);
    }
}



void AxonSodium_neon(cellStateV &st, uint idx){

    //opt for division by constant
    auto const five_five = 1.0f/5.5f; //no change in area by synthesizer on V7
    auto const five_eight = -1.0f/5.8f; //no change in area by synthesizer on V7
    auto const thirty_three = 1.0f/33.0f;  //no change in area by synthesizer on V7

    //Get inputs
    auto const prevV_axon = vld1q_f32(st.Prev_V_Axon.data()+idx);
    auto const prevSodium_h_a = vld1q_f32(st.Sodium_h_a.data()+idx);

    // Update axonal Na components
    // NOTE: current has shortened inactivation to account for high
    // firing frequencies in axon hillock
    auto const exp_m_inf_a = exp_ps(vmulq_f32(vaddq_f32(vnegq_f32(prevV_axon),oneVec(-30.0f)),oneVec(five_five)));
    auto const m_inf_a = vrecpeq_f32(vaddq_f32(exp_m_inf_a,oneVec(1.0f)));
    auto const exp_h_inf_a = exp_ps(vmulq_f32(vaddq_f32(vnegq_f32(prevV_axon),oneVec(-60.0f)),oneVec(five_eight)));
    auto const h_inf_a = vrecpeq_f32(vaddq_f32(exp_h_inf_a,oneVec(1.0f)));
    auto const exp_tau_h_a = exp_ps(vmulq_f32(vaddq_f32(vnegq_f32(prevV_axon),oneVec(-40.0f)),oneVec(thirty_three)));
    auto const tau_h_a = vmulq_f32(exp_tau_h_a,oneVec(1.5f));
    auto const dh_dt_a = vmulq_f32(vaddq_f32(vnegq_f32(prevSodium_h_a),h_inf_a),vrecpeq_f32(tau_h_a));
    auto const m_a_local = m_inf_a;
    auto const h_a_local = vaddq_f32(vmulq_f32(oneVec(DELTA),dh_dt_a),prevSodium_h_a);

    //Put result
    vst1q_f32(st.Sodium_m_a.data()+idx,m_a_local);
    vst1q_f32(st.Sodium_h_a.data()+idx,h_a_local);
}



void AxonPotassium_neon(cellStateV &st, uint idx){
    //opt for division by constant
    auto const ten = 0.1; //no change in area by synthesizer on V7

    //Get inputs
    auto const prevV_axon = vld1q_f32(st.Prev_V_Axon.data()+idx);
    auto const prevPotassium_x_a = vld1q_f32(st.Potassium_x_a.data()+idx);

    // D'ANGELO 2001 -- Voltage-dependent potassium
    auto const exp_alpha_x_a = exp_ps(vmulq_f32(vnegq_f32(vaddq_f32(prevV_axon,oneVec(25.0f))),oneVec(ten)));
    auto const num_alpha_x_a = vmulq_f32(vaddq_f32(prevV_axon,oneVec(25.0f)),oneVec(0.13f));
    auto const den_alpha_x_a = vrecpeq_f32(vaddq_f32(vnegq_f32(exp_alpha_x_a),oneVec(1.0f)));
    auto const alpha_x_a = vmulq_f32(num_alpha_x_a,den_alpha_x_a);    

    auto const beta_x_a  = vmulq_f32(exp_ps(vmulq_f32(vaddq_f32(prevV_axon,oneVec(35.0f)),oneVec(-0.0125f))),oneVec(1.69f));
    auto const temp_x_a_local = vaddq_f32(vnegq_f32(vmulq_f32(vaddq_f32(alpha_x_a,beta_x_a),oneVec(DELTA))),oneVec(1.0f));
    auto const x_a_local = vaddq_f32(vmulq_f32(temp_x_a_local,prevPotassium_x_a),vmulq_f32(alpha_x_a,oneVec(DELTA)));

    //Put result
    vst1q_f32(st.Potassium_x_a.data()+idx,x_a_local);
}



void AxonCurrVolt_neon(cellStateV &st, uint idx){

    auto constexpr div_GINT_P2 = (G_INT / P2);

    //opt for division by constant
    auto const inv_CM= 1/C_M;

    //Get inputs
    auto const prevV_soma = vld1q_f32(st.Prev_V_Soma.data()+idx);
    auto const prevV_axon = vld1q_f32(st.Prev_V_Axon.data()+idx);
    auto const m_a = vld1q_f32(st.Sodium_m_a.data()+idx);
    auto const h_a = vld1q_f32(st.Sodium_h_a.data()+idx);
    auto const x_a = vld1q_f32(st.Potassium_x_a.data()+idx);

    // AXONAL CURRENTS
    // Sodium
    auto const m_a_exp_3 = vmulq_f32(vmulq_f32(m_a,m_a),m_a);
    auto const I_Na_a = vmulq_f32(vmulq_f32(vmulq_f32(vaddq_f32(prevV_axon,oneVec(-V_NA)),h_a),m_a_exp_3),oneVec(G_NA_A));
    // Leak
    auto const I_la = vmulq_f32(vaddq_f32(prevV_axon,oneVec(-V_L)),oneVec(G_LA));
    // Soma-axon interaction current I_sa
    auto const I_sa = vmulq_f32(vsubq_f32(prevV_axon,prevV_soma),oneVec(div_GINT_P2));
    // Potassium (transient)
    auto const x_a_exp_4 = vmulq_f32(vmulq_f32(vmulq_f32(x_a,x_a),x_a),x_a);
    auto const I_K_a = vmulq_f32(vmulq_f32(vaddq_f32(prevV_axon,oneVec(-V_K)),x_a_exp_4),oneVec(G_K_A));

    auto const sumatory = vaddq_f32(vaddq_f32(vaddq_f32(I_K_a,I_sa),I_la),I_Na_a);
    auto const dVa_dt = vmulq_f32(vnegq_f32(sumatory),oneVec(inv_CM));

    //Put result
    auto const newVAxon = vaddq_f32(vmulq_f32(oneVec(DELTA),dVa_dt),prevV_axon);
    vst1q_f32(st.V_Axon.data()+idx,newVAxon);
}

void ComputeNetwork_neon(cellStateV &s, SimConfig &conf){
   const auto n=conf.local_population;
   std::thread th(CompDend_neon,std::ref(s),n);
   //CompDend_neon(s,n);
   CompSoma_neon(s,n);
   CompAxon_neon(s,n);
   th.join();
   swapVec(s);
}
