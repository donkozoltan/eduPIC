//-------------------------------------------------------------------//
//         eduPIC : educational 1d3v PIC/MCC simulation code         //
//           version 1.0, release date: March 16, 2021               //
//                       :) Share & enjoy :)                         //
//-------------------------------------------------------------------//
// When you use this code, you are required to acknowledge the       //
// authors by citing the paper:                                      //
// Z. Donko, A. Derzsi, M. Vass, B. Horvath, S. Wilczek              //
// B. Hartmann, P. Hartmann:                                         //
// "eduPIC: an introductory particle based  code for radio-frequency //
// plasma simulation"                                                //
// Plasma Sources Science and Technology, vol XXX, pp. XXX (2021)    //
//-------------------------------------------------------------------//
// Disclaimer: The eduPIC (educational Particle-in-Cell/Monte Carlo  //
// Collisions simulation code), Copyright (C) 2021                   //
// Zoltan Donko et al. is free software: you can redistribute it     //
// and/or modify it under the terms of the GNU General Public License//
// as published by the Free Software Foundation, version 3.          //
// This program is distributed in the hope that it will be useful,   //
// but WITHOUT ANY WARRANTY; without even the implied warranty of    //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU  //
// General Public License for more details at                        //
// https://www.gnu.org/licenses/gpl-3.0.html.                        //
//-------------------------------------------------------------------//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <cmath>
#include <ctime>
#include <random>

using namespace::std;

// constants

const double     PI             = 3.141592653589793;          // mathematical constant Pi
const double     TWO_PI         = 2.0 * PI;                   // two times Pi
const double     E_CHARGE       = 1.60217662e-19;             // electron charge [C]
const double     EV_TO_J        = E_CHARGE;                   // eV <-> Joule conversion factor
const double     E_MASS         = 9.10938356e-31;             // mass of electron [kg]
const double     AR_MASS        = 6.63352090e-26;             // mass of argon atom [kg]
const double     MU_ARAR        = AR_MASS / 2.0;              // reduced mass of two argon atoms [kg]
const double     K_BOLTZMANN    = 1.38064852e-23;             // Boltzmann's constant [J/K]
const double     EPSILON0       = 8.85418781e-12;             // permittivity of free space [F/m]

// simulation parameters

const int        N_G            = 400;                        // number of grid points
const int        N_T            = 4000;                       // time steps within an RF period
const double     FREQUENCY      = 13.56e6;                    // driving frequency [Hz]
const double     VOLTAGE        = 250.0;                      // voltage amplitude [V]
const double     L              = 0.025;                      // electrode gap [m]
const double     PRESSURE       = 10.0;                       // gas pressure [Pa]
const double     TEMPERATURE    = 350.0;                      // background gas temperature [K]
const double     WEIGHT         = 7.0e4;                      // weight of superparticles
const double     ELECTRODE_AREA = 1.0e-4;                     // (fictive) electrode area [m^2]
const int        N_INIT         = 1000;                       // number of initial electrons and ions

// additional (derived) constants

const double     PERIOD         = 1.0 / FREQUENCY;                           // RF period length [s]
const double     DT_E           = PERIOD / (double)(N_T);                    // electron time step [s]
const int        N_SUB          = 20;                                        // ions move only in these cycles (subcycling)
const double     DT_I           = N_SUB * DT_E;                              // ion time step [s]
const double     DX             = L / (double)(N_G - 1);                     // spatial grid division [m]
const double     INV_DX         = 1.0 / DX;                                  // inverse of spatial grid size [1/m]
const double     GAS_DENSITY    = PRESSURE / (K_BOLTZMANN * TEMPERATURE);    // background gas density [1/m^3]
const double     OMEGA          = TWO_PI * FREQUENCY;                        // angular frequency [rad/s]

// electron and ion cross sections

const int        N_CS           = 5;                          // total number of processes / cross sections
const int        E_ELA          = 0;                          // process identifier: electron/elastic
const int        E_EXC          = 1;                          // process identifier: electron/excitation
const int        E_ION          = 2;                          // process identifier: electron/ionization
const int        I_ISO          = 3;                          // process identifier: ion/elastic/isotropic
const int        I_BACK         = 4;                          // process identifier: ion/elastic/backscattering
const double     E_EXC_TH       = 11.5;                       // electron impact excitation threshold [eV]
const double     E_ION_TH       = 15.8;                       // electron impact ionization threshold [eV]
const int        CS_RANGES      = 1000000;                    // number of entries in cross section arrays
const double     DE_CS          = 0.001;                      // energy division in cross section arrays [eV]
typedef float    cross_section[CS_RANGES];                    // cross section array
cross_section    sigma[N_CS];                                 // set of cross section arrays
cross_section    sigma_tot_e;                                 // total macroscopic cross section of electrons
cross_section    sigma_tot_i;                                 // total macroscopic cross section of ions

// particle coordinates

const int        MAX_N_P = 1000000;                           // maximum number of particles (electrons / ions)
typedef double   particle_vector[MAX_N_P];                    // array for particle properties
int              N_e = 0;                                     // number of electrons
int              N_i = 0;                                     // number of ions
particle_vector  x_e, vx_e, vy_e, vz_e;                       // coordinates of electrons (one spatial, three velocity components)
particle_vector  x_i, vx_i, vy_i, vz_i;                       // coordinates of ions (one spatial, three velocity components)

typedef double   xvector[N_G];                                // array for quantities defined at gird points
xvector          efield,pot;                                  // electric field and potential
xvector          e_density,i_density;                         // electron and ion densities
xvector          cumul_e_density,cumul_i_density;             // cumulative densities

typedef unsigned long long int Ullong;                        // compact name for 64 bit unsigned integer
Ullong       N_e_abs_pow  = 0;                                // counter for electrons absorbed at the powered electrode
Ullong       N_e_abs_gnd  = 0;                                // counter for electrons absorbed at the grounded electrode
Ullong       N_i_abs_pow  = 0;                                // counter for ions absorbed at the powered electrode
Ullong       N_i_abs_gnd  = 0;                                // counter for ions absorbed at the grounded electrode

// electron energy probability function

const int    N_EEPF  = 2000;                                 // number of energy bins in Electron Energy Probability Function (EEPF)
const double DE_EEPF = 0.05;                                 // resolution of EEPF [eV]
typedef double eepf_vector[N_EEPF];                          // array for EEPF
eepf_vector eepf     = {0.0};                                // time integrated EEPF in the center of the plasma

// ion flux-energy distributions

const int    N_IFED   = 200;                                 // number of energy bins in Ion Flux-Energy Distributions (IFEDs)
const double DE_IFED  = 1.0;                                 // resolution of IFEDs [eV]
typedef int  ifed_vector[N_IFED];                            // array for IFEDs
ifed_vector  ifed_pow = {0};                                 // IFED at the powered electrode
ifed_vector  ifed_gnd = {0};                                 // IFED at the grounded electrode
double       mean_i_energy_pow;                              // mean ion energy at the powered electrode
double       mean_i_energy_gnd;                              // mean ion energy at the grounded electrode

// spatio-temporal (XT) distributions

const int N_BIN                     = 20;                    // number of time steps binned for the XT distributions
const int N_XT                      = N_T / N_BIN;           // number of spatial bins for the XT distributions
typedef double xt_distr[N_G][N_XT];                          // array for XT distributions (decimal numbers)
xt_distr pot_xt                     = {0.0};                 // XT distribution of the potential
xt_distr efield_xt                  = {0.0};                 // XT distribution of the electric field
xt_distr ne_xt                      = {0.0};                 // XT distribution of the electron density
xt_distr ni_xt                      = {0.0};                 // XT distribution of the ion density
xt_distr ue_xt                      = {0.0};                 // XT distribution of the mean electron velocity
xt_distr ui_xt                      = {0.0};                 // XT distribution of the mean ion velocity
xt_distr je_xt                      = {0.0};                 // XT distribution of the electron current density
xt_distr ji_xt                      = {0.0};                 // XT distribution of the ion current density
xt_distr powere_xt                  = {0.0};                 // XT distribution of the electron powering (power absorption) rate
xt_distr poweri_xt                  = {0.0};                 // XT distribution of the ion powering (power absorption) rate
xt_distr meanee_xt                  = {0.0};                 // XT distribution of the mean electron energy
xt_distr meanei_xt                  = {0.0};                 // XT distribution of the mean ion energy
xt_distr counter_e_xt               = {0.0};                 // XT counter for electron properties
xt_distr counter_i_xt               = {0.0};                 // XT counter for ion properties
xt_distr ioniz_rate_xt              = {0.0};                 // XT distribution of the ionisation rate

double   mean_energy_accu_center    = 0;                     // mean electron energy accumulator in the center of the gap
Ullong   mean_energy_counter_center = 0;                     // mean electron energy counter in the center of the gap
Ullong   N_e_coll                   = 0;                     // counter for electron collisions
Ullong   N_i_coll                   = 0;                     // counter for ion collisions
double   Time;                                               // total simulated time (from the beginning of the simulation)
int      cycle,no_of_cycles,cycles_done;                     // current cycle and total cycles in the run, cycles completed
int      arg1;                                               // used for reading command line arguments
char     st0[80];                                            // used for reading command line arguments
FILE     *datafile;                                          // used for saving data
bool     measurement_mode;                                   // flag that controls measurements and data saving

//---------------------------------------------------------------------------//
// C++ Mersenne Twister 19937 generator                                      //
// R01(MTgen) will genarate uniform distribution over [0,1) interval         //
// RMB(MTgen) will generate Maxwell-Boltzmann distribution (of gas atoms)    //
//---------------------------------------------------------------------------//

std::random_device rd{}; 
std::mt19937 MTgen(rd());
std::uniform_real_distribution<> R01(0.0, 1.0);
std::normal_distribution<> RMB(0.0,sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS));

//----------------------------------------------------------------------------//
//  electron cross sections: A V Phelps & Z Lj Petrovic, PSST 8 R21 (1999)    //
//----------------------------------------------------------------------------//

void set_electron_cross_sections_ar(void){
    int    i;
    double en,qmel,qexc,qion;
    
    printf(">> eduPIC: Setting e- / Ar cross sections\n");
    for(i=0; i<CS_RANGES; i++){
        if (i == 0) {en = DE_CS;} else {en = DE_CS * i;}                            // electron energy
        qmel = fabs(6.0 / pow(1.0 + (en/0.1) + pow(en/0.6,2.0), 3.3)
                    - 1.1 * pow(en, 1.4) / (1.0 + pow(en/15.0, 1.2)) / sqrt(1.0 + pow(en/5.5, 2.5) + pow(en/60.0, 4.1)))
        + 0.05 / pow(1.0 + en/10.0, 2.0) + 0.01 * pow(en, 3.0) / (1.0 + pow(en/12.0, 6.0));
        if (en > E_EXC_TH)
            qexc = 0.034 * pow(en-11.5, 1.1) * (1.0 + pow(en/15.0, 2.8)) / (1.0 + pow(en/23.0, 5.5))
            + 0.023 * (en-11.5) / pow(1.0 + en/80.0, 1.9);
        else
            qexc = 0;
        if (en > E_ION_TH)
            qion = 970.0 * (en-15.8) / pow(70.0 + en, 2.0) + 0.06 * pow(en-15.8, 2.0) * exp(-en/9);
        else
            qion = 0;
        sigma[E_ELA][i] = qmel * 1.0e-20;       // cross section for e- / Ar elastic collision
        sigma[E_EXC][i] = qexc * 1.0e-20;       // cross section for e- / Ar excitation
        sigma[E_ION][i] = qion * 1.0e-20;       // cross section for e- / Ar ionization
    }
}

//------------------------------------------------------------------------------//
//  ion cross sections: A. V. Phelps, J. Appl. Phys. 76, 747 (1994)             //
//------------------------------------------------------------------------------//

void set_ion_cross_sections_ar(void){
    int    i;
    double e_com,e_lab,qmom,qback,qiso;
    
    printf(">> eduPIC: Setting Ar+ / Ar cross sections\n");
    for(i=0; i<CS_RANGES; i++){
        if (i == 0) {e_com = DE_CS;} else {e_com = DE_CS * i;}             // ion energy in the center of mass frame of reference
        e_lab = 2.0 * e_com;                                               // ion energy in the laboratory frame of reference
        qmom  = 1.15e-18 * pow(e_lab,-0.1) * pow(1.0 + 0.015 / e_lab, 0.6);
        qiso  = 2e-19 * pow(e_lab,-0.5) / (1.0 + e_lab) + 3e-19 * e_lab / pow(1.0 + e_lab / 3.0, 2.0);
        qback = (qmom-qiso) / 2.0;
        sigma[I_ISO][i]  = qiso;             // cross section for Ar+ / Ar isotropic part of elastic scattering
        sigma[I_BACK][i] = qback;            // cross section for Ar+ / Ar backward elastic scattering
    }
}

//----------------------------------------------------------------------//
//  calculation of total cross sections for electrons and ions          //
//----------------------------------------------------------------------//

void calc_total_cross_sections(void){
    int i;
    
    for(i=0; i<CS_RANGES; i++){
        sigma_tot_e[i] = (sigma[E_ELA][i] + sigma[E_EXC][i] + sigma[E_ION][i]) * GAS_DENSITY;   // total macroscopic cross section of electrons
        sigma_tot_i[i] = (sigma[I_ISO][i] + sigma[I_BACK][i]) * GAS_DENSITY;                    // total macroscopic cross section of ions
    }
}

//----------------------------------------------------------------------//
//  test of cross sections for electrons and ions                       //
//----------------------------------------------------------------------//

void test_cross_sections(void){
    FILE  * f;
    int   i,j;
    
    f = fopen("cross_sections.dat","w");        // cross sections saved in data file: cross_sections.dat
    for(i=0; i<CS_RANGES; i++){
        fprintf(f,"%12.4f ",i*DE_CS);
        for(j=0; j<N_CS; j++) fprintf(f,"%14e ",sigma[j][i]);
        fprintf(f,"\n");
    }
    fclose(f);
}

//---------------------------------------------------------------------//
// find upper limit of collision frequencies                           //
//---------------------------------------------------------------------//

double max_electron_coll_freq (void){
    int i;
    double e,v,nu,nu_max;
    nu_max = 0;
    for(i=0; i<CS_RANGES; i++){
        e  = i * DE_CS;
        v  = sqrt(2.0 * e * EV_TO_J / E_MASS);
        nu = v * sigma_tot_e[i];
        if (nu > nu_max) {nu_max = nu;}
    }
    return nu_max;
}

double max_ion_coll_freq (void){
    int i;
    double e,g,nu,nu_max;
    nu_max = 0;
    for(i=0; i<CS_RANGES; i++){
        e  = i * DE_CS;
        g  = sqrt(2.0 * e * EV_TO_J / MU_ARAR);
        nu = g * sigma_tot_i[i];
        if (nu > nu_max) nu_max = nu;
    }
    return nu_max;
}

//----------------------------------------------------------------------//
// initialization of the simulation by placing a given number of        //
// electrons and ions at random positions between the electrodes        //
//----------------------------------------------------------------------//

void init(int nseed){
    int i;
    
    for (i=0; i<nseed; i++){
        x_e[i]  = L * R01(MTgen);               // initial random position of the electron
        vx_e[i] = 0; vy_e[i] = 0; vz_e[i] = 0;  // initial velocity components of the electron
        x_i[i]  = L * R01(MTgen);               // initial random position of the ion
        vx_i[i] = 0; vy_i[i] = 0; vz_i[i] = 0;  // initial velocity components of the ion
    }
    N_e = nseed;    // initial number of electrons
    N_i = nseed;    // initial number of ions
}

//----------------------------------------------------------------------//
// e / Ar collision  (cold gas approximation)                           //
//----------------------------------------------------------------------//

void collision_electron (double xe, double *vxe, double *vye, double *vze, int eindex){
    const double F1 = E_MASS  / (E_MASS + AR_MASS);
    const double F2 = AR_MASS / (E_MASS + AR_MASS);
    double t0,t1,t2,rnd;
    double g,g2,gx,gy,gz,wx,wy,wz,theta,phi;
    double chi,eta,chi2,eta2,sc,cc,se,ce,st,ct,sp,cp,energy,e_sc,e_ej;
    
    // calculate relative velocity before collision & velocity of the centre of mass
    
    gx = (*vxe);
    gy = (*vye);
    gz = (*vze);
    g  = sqrt(gx * gx + gy * gy + gz * gz);
    wx = F1 * (*vxe);
    wy = F1 * (*vye);
    wz = F1 * (*vze);
    
    // find Euler angles
    
    if (gx == 0) {theta = 0.5 * PI;}
    else {theta = atan2(sqrt(gy * gy + gz * gz),gx);}
    if (gy == 0) {
        if (gz > 0){phi = 0.5 * PI;} else {phi = - 0.5 * PI;}
    } else {phi = atan2(gz, gy);}
    st  = sin(theta);
    ct  = cos(theta);
    sp  = sin(phi);
    cp  = cos(phi);
    
    // choose the type of collision based on the cross sections
    // take into account energy loss in inelastic collisions
    // generate scattering and azimuth angles
    // in case of ionization handle the 'new' electron
    
    t0   =     sigma[E_ELA][eindex];
    t1   = t0 +sigma[E_EXC][eindex];
    t2   = t1 +sigma[E_ION][eindex];
    rnd  = R01(MTgen);
    if (rnd < (t0/t2)){                              // elastic scattering
        chi = acos(1.0 - 2.0 * R01(MTgen));          // isotropic scattering
        eta = TWO_PI * R01(MTgen);                   // azimuthal angle
    } else if (rnd < (t1/t2)){                       // excitation
        energy = 0.5 * E_MASS * g * g;               // electron energy
        energy = fabs(energy - E_EXC_TH * EV_TO_J);  // subtract energy loss for excitation
        g   = sqrt(2.0 * energy / E_MASS);           // relative velocity after energy loss
        chi = acos(1.0 - 2.0 * R01(MTgen));          // isotropic scattering
        eta = TWO_PI * R01(MTgen);                   // azimuthal angle
    } else {                                         // ionization
        energy = 0.5 * E_MASS * g * g;               // electron energy
        energy = fabs(energy - E_ION_TH * EV_TO_J);  // subtract energy loss of ionization
        e_ej  = 10.0 * tan(R01(MTgen) * atan(energy/EV_TO_J / 20.0)) * EV_TO_J; // energy of the ejected electron
        e_sc = fabs(energy - e_ej);                  // energy of scattered electron after the collision
        g    = sqrt(2.0 * e_sc / E_MASS);            // relative velocity of scattered electron
        g2   = sqrt(2.0 * e_ej / E_MASS);            // relative velocity of ejected electron
        chi  = acos(sqrt(e_sc / energy));            // scattering angle for scattered electron
        chi2 = acos(sqrt(e_ej / energy));            // scattering angle for ejected electrons
        eta  = TWO_PI * R01(MTgen);                  // azimuthal angle for scattered electron
        eta2 = eta + PI;                             // azimuthal angle for ejected electron
        sc  = sin(chi2);
        cc  = cos(chi2);
        se  = sin(eta2);
        ce  = cos(eta2);
        gx  = g2 * (ct * cc - st * sc * ce);
        gy  = g2 * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
        gz  = g2 * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
        x_e[N_e]  = xe;                              // add new electron
        vx_e[N_e] = wx + F2 * gx;
        vy_e[N_e] = wy + F2 * gy;
        vz_e[N_e] = wz + F2 * gz;
        N_e++;
        x_i[N_i]  = xe;                              // add new ion
        vx_i[N_i] = RMB(MTgen);                      // velocity is sampled from background thermal distribution
        vy_i[N_i] = RMB(MTgen);
        vz_i[N_i] = RMB(MTgen);
        N_i++;
    }
    
    // scatter the primary electron
    
    sc = sin(chi);
    cc = cos(chi);
    se = sin(eta);
    ce = cos(eta);
    
    // compute new relative velocity:
    
    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
    
    // post-collision velocity of the colliding electron
    
    (*vxe) = wx + F2 * gx;
    (*vye) = wy + F2 * gy;
    (*vze) = wz + F2 * gz;
}

//----------------------------------------------------------------------//
// Ar+ / Ar collision                                                   //
//----------------------------------------------------------------------//

void collision_ion (double *vx_1, double *vy_1, double *vz_1,
                    double *vx_2, double *vy_2, double *vz_2, int e_index){
    double   g,gx,gy,gz,wx,wy,wz,rnd;
    double   theta,phi,chi,eta,st,ct,sp,cp,sc,cc,se,ce,t1,t2;
    
    // calculate relative velocity before collision
    // random Maxwellian target atom already selected (vx_2,vy_2,vz_2 velocity components of target atom come with the call)
    
    gx = (*vx_1)-(*vx_2);
    gy = (*vy_1)-(*vy_2);
    gz = (*vz_1)-(*vz_2);
    g  = sqrt(gx * gx + gy * gy + gz * gz);
    wx = 0.5 * ((*vx_1) + (*vx_2));
    wy = 0.5 * ((*vy_1) + (*vy_2));
    wz = 0.5 * ((*vz_1) + (*vz_2));
    
    // find Euler angles
    
    if (gx == 0) {theta = 0.5 * PI;} else {theta = atan2(sqrt(gy * gy + gz * gz),gx);}
    if (gy == 0) {
        if (gz > 0){phi = 0.5 * PI;} else {phi = - 0.5 * PI;}
    } else {phi = atan2(gz, gy);}
    
    // determine the type of collision based on cross sections and generate scattering angle
    
    t1  =      sigma[I_ISO][e_index];
    t2  = t1 + sigma[I_BACK][e_index];
    rnd = R01(MTgen);
    if  (rnd < (t1 /t2)){                        // isotropic scattering
        chi = acos(1.0 - 2.0 * R01(MTgen));      // scattering angle
    } else {                                     // backward scattering
        chi = PI;                                // scattering angle
    }
    eta = TWO_PI * R01(MTgen);                   // azimuthal angle
    sc  = sin(chi);
    cc  = cos(chi);
    se  = sin(eta);
    ce  = cos(eta);
    st  = sin(theta);
    ct  = cos(theta);
    sp  = sin(phi);
    cp  = cos(phi);
    
    // compute new relative velocity
    
    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
    
    // post-collision velocity of the ion
    
    (*vx_1) = wx + 0.5 * gx;
    (*vy_1) = wy + 0.5 * gy;
    (*vz_1) = wz + 0.5 * gz;
}

//-----------------------------------------------------------------//
// solve Poisson equation (Thomas algorithm)                       //
//-----------------------------------------------------------------//

void solve_Poisson (xvector rho1, double tt){
    const double A =  1.0;
    const double B = -2.0;
    const double C =  1.0;
    const double S = 1.0 / (2.0 * DX);
    const double ALPHA = -DX * DX / EPSILON0;
    xvector      g, w, f;
    int          i;
    
    // apply potential to the electrodes - boundary conditions
    
    pot[0]     = VOLTAGE * cos(OMEGA * tt);         // potential at the powered electrode
    pot[N_G-1] = 0.0;                               // potential at the grounded electrode
    
    // solve Poisson equation
    
    for(i=1; i<=N_G-2; i++) f[i] = ALPHA * rho1[i];
    f[1] -= pot[0];
    f[N_G-2] -= pot[N_G-1];
    w[1] = C/B;
    g[1] = f[1]/B;
    for(i=2; i<=N_G-2; i++){
        w[i] = C / (B - A * w[i-1]);
        g[i] = (f[i] - A * g[i-1]) / (B - A * w[i-1]);
    }
    pot[N_G-2] = g[N_G-2];
    for (i=N_G-3; i>0; i--) pot[i] = g[i] - w[i] * pot[i+1];            // potential at the grid points between the electrodes
    
    // compute electric field
    
    for(i=1; i<=N_G-2; i++) efield[i] = (pot[i-1] - pot[i+1]) * S;      // electric field at the grid points between the electrodes
    efield[0]     = (pot[0]     - pot[1])     * INV_DX - rho1[0]     * DX / (2.0 * EPSILON0);   // powered electrode
    efield[N_G-1] = (pot[N_G-2] - pot[N_G-1]) * INV_DX + rho1[N_G-1] * DX / (2.0 * EPSILON0);   // grounded electrode
}

//---------------------------------------------------------------------//
// simulation of one radiofrequency cycle                              //
//---------------------------------------------------------------------//

void do_one_cycle (void){
    const double DV       = ELECTRODE_AREA * DX;
    const double FACTOR_W = WEIGHT / DV;
    const double FACTOR_E = DT_E / E_MASS * E_CHARGE;
    const double FACTOR_I = DT_I / AR_MASS * E_CHARGE;
    const double MIN_X    = 0.45 * L;                       // min. position for EEPF collection
    const double MAX_X    = 0.55 * L;                       // max. position for EEPF collection
    int      k, t, p, energy_index;
    double   g, g_sqr, gx, gy, gz, vx_a, vy_a, vz_a, e_x, energy, nu, p_coll, v_sqr, velocity;
    double   mean_v, c0, c1, c2, rate;
    bool     out;
    xvector  rho;
    int      t_index;
    
    for (t=0; t<N_T; t++){          // the RF period is divided into N_T equal time intervals (time step DT_E)
        Time += DT_E;               // update of the total simulated time
        t_index = t / N_BIN;        // index for XT distributions
        
        // step 1: compute densities at grid points
        
        for(p=0; p<N_G; p++) e_density[p] = 0;                             // electron density - computed in every time step
        for(k=0; k<N_e; k++){
            c0 = x_e[k] * INV_DX;
            p  = int(c0);
            e_density[p]   += (p + 1 - c0) * FACTOR_W;
            e_density[p+1] += (c0 - p) * FACTOR_W;
        }
        e_density[0]     *= 2.0;
        e_density[N_G-1] *= 2.0;
        for(p=0; p<N_G; p++) cumul_e_density[p] += e_density[p];
        
        if ((t % N_SUB) == 0) {                                            // ion density - computed in every N_SUB-th time steps (subcycling)
            for(p=0; p<N_G; p++) i_density[p] = 0;
            for(k=0; k<N_i; k++){
                c0 = x_i[k] * INV_DX;
                p  = int(c0);
                i_density[p]   += (p + 1 - c0) * FACTOR_W;  
                i_density[p+1] += (c0 - p) * FACTOR_W;
            }
            i_density[0]     *= 2.0;
            i_density[N_G-1] *= 2.0;
        }
        for(p=0; p<N_G; p++) cumul_i_density[p] += i_density[p];
        
        // step 2: solve Poisson equation
        
        for(p=0; p<N_G; p++) rho[p] = E_CHARGE * (i_density[p] - e_density[p]);  // get charge density
        solve_Poisson(rho,Time);                                                 // compute potential and electric field
        
        // steps 3 & 4: move particles according to electric field interpolated to particle positions
        
        for(k=0; k<N_e; k++){                       // move all electrons in every time step
            c0  = x_e[k] * INV_DX;
            p   = int(c0);
            c1  = p + 1.0 - c0;
            c2  = c0 - p;
            e_x = c1 * efield[p] + c2 * efield[p+1];
            
            if (measurement_mode) {
                
                // measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'
                
                mean_v = vx_e[k] - 0.5 * e_x * FACTOR_E;
                counter_e_xt[p][t_index]   += c1;
                counter_e_xt[p+1][t_index] += c2;
                ue_xt[p][t_index]   += c1 * mean_v;
                ue_xt[p+1][t_index] += c2 * mean_v;
                v_sqr  = mean_v * mean_v + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k];
                energy = 0.5 * E_MASS * v_sqr / EV_TO_J;
                meanee_xt[p][t_index]   += c1 * energy;
                meanee_xt[p+1][t_index] += c2 * energy;
                energy_index = min( int(energy / DE_CS + 0.5), CS_RANGES-1);
                velocity = sqrt(v_sqr);
                rate = sigma[E_ION][energy_index] * velocity * DT_E * GAS_DENSITY;
                ioniz_rate_xt[p][t_index]   += c1 * rate;
                ioniz_rate_xt[p+1][t_index] += c2 * rate;

                // measure EEPF in the center
                
                if ((MIN_X < x_e[k]) && (x_e[k] < MAX_X)){
                    energy_index = (int)(energy / DE_EEPF);
                    if (energy_index < N_EEPF) {eepf[energy_index] += 1.0;}
                    mean_energy_accu_center += energy;
                    mean_energy_counter_center++;
                }
            }
            
            // update velocity and position
            
            vx_e[k] -= e_x * FACTOR_E;
            x_e[k]  += vx_e[k] * DT_E;
        }
        
        if ((t % N_SUB) == 0) {                       // move all ions in every N_SUB-th time steps (subcycling)
            for(k=0; k<N_i; k++){
                c0  = x_i[k] * INV_DX;
                p   = int(c0);
                c1  = p + 1 - c0;
                c2  = c0 - p;
                e_x = c1 * efield[p] + c2 * efield[p+1];
                
                if (measurement_mode) {
                    
                    // measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'

                    mean_v = vx_i[k] + 0.5 * e_x * FACTOR_I;
                    counter_i_xt[p][t_index]   += c1;
                    counter_i_xt[p+1][t_index] += c2;
                    ui_xt[p][t_index]   += c1 * mean_v;
                    ui_xt[p+1][t_index] += c2 * mean_v;
                    v_sqr  = mean_v * mean_v + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    meanei_xt[p][t_index]   += c1 * energy;
                    meanei_xt[p+1][t_index] += c2 * energy;
                }
                
                // update velocity and position and accumulate absorbed energy
                
                vx_i[k] += e_x * FACTOR_I;
                x_i[k]  += vx_i[k] * DT_I;
            }
        }
        
        // step 5: check boundaries
        
        k = 0;
        while(k < N_e) {    // check boundaries for all electrons in every time step
            out = false;
            if (x_e[k] < 0) {N_e_abs_pow++; out = true;}    // the electron is out at the powered electrode
            if (x_e[k] > L) {N_e_abs_gnd++; out = true;}    // the electron is out at the grounded electrode
            if (out) {                                      // remove the electron, if out
                x_e [k] = x_e [N_e-1];
                vx_e[k] = vx_e[N_e-1];
                vy_e[k] = vy_e[N_e-1];
                vz_e[k] = vz_e[N_e-1];
                N_e--;
            } else k++;
        }
        
        if ((t % N_SUB) == 0) {   // check boundaries for all ions in every N_SUB-th time steps (subcycling)
            k = 0;
            while(k < N_i) {
                out = false;
                if (x_i[k] < 0) {       // the ion is out at the powered electrode
                    N_i_abs_pow++;
                    out    = true;
                    v_sqr  = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS *  v_sqr/ EV_TO_J;
                    energy_index = (int)(energy / DE_IFED);
                    if (energy_index < N_IFED) {ifed_pow[energy_index]++;}       // save IFED at the powered electrode
                }
                if (x_i[k] > L) {       // the ion is out at the grounded electrode
                    N_i_abs_gnd++;
                    out    = true;
                    v_sqr  = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    energy_index = (int)(energy / DE_IFED);
                    if (energy_index < N_IFED) {ifed_gnd[energy_index]++;}        // save IFED at the grounded electrode
                }
                if (out) {  // delete the ion, if out
                    x_i [k] = x_i [N_i-1];
                    vx_i[k] = vx_i[N_i-1];
                    vy_i[k] = vy_i[N_i-1];
                    vz_i[k] = vz_i[N_i-1];
                    N_i--;
                } else k++;
            }
        }
        
        // step 6: collisions
        
        for (k=0; k<N_e; k++){                              // checking for occurrence of a collision for all electrons in every time step
            v_sqr = vx_e[k] * vx_e[k] + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k];
            velocity = sqrt(v_sqr);
            energy   = 0.5 * E_MASS * v_sqr / EV_TO_J;
            energy_index = min( int(energy / DE_CS + 0.5), CS_RANGES-1);
            nu = sigma_tot_e[energy_index] * velocity;
            p_coll = 1 - exp(- nu * DT_E);                  // collision probability for electrons
            if (R01(MTgen) < p_coll) {                      // electron collision takes place
                collision_electron(x_e[k], &vx_e[k], &vy_e[k], &vz_e[k], energy_index);
                N_e_coll++;
            }
        }
        
        if ((t % N_SUB) == 0) {                             // checking for occurrence of a collision for all ions in every N_SUB-th time steps (subcycling)
            for (k=0; k<N_i; k++){
                vx_a = RMB(MTgen);                          // pick velocity components of a random target gas atom
                vy_a = RMB(MTgen);
                vz_a = RMB(MTgen);
                gx   = vx_i[k] - vx_a;                       // compute the relative velocity of the collision partners
                gy   = vy_i[k] - vy_a;
                gz   = vz_i[k] - vz_a;
                g_sqr = gx * gx + gy * gy + gz * gz;
                g = sqrt(g_sqr);
                energy = 0.5 * MU_ARAR * g_sqr / EV_TO_J;
                energy_index = min( int(energy / DE_CS + 0.5), CS_RANGES-1);
                nu = sigma_tot_i[energy_index] * g;
                p_coll = 1 - exp(- nu * DT_I);              // collision probability for ions
                if (R01(MTgen)< p_coll) {                   // ion collision takes place
                    collision_ion (&vx_i[k], &vy_i[k], &vz_i[k], &vx_a, &vy_a, &vz_a, energy_index);
                    N_i_coll++;
                }
            }
        }
        
        if (measurement_mode) {
            
            // collect 'xt' data from the grid
            
            for (p=0; p<N_G; p++) {
                pot_xt   [p][t_index] += pot[p];
                efield_xt[p][t_index] += efield[p];
                ne_xt    [p][t_index] += e_density[p];
                ni_xt    [p][t_index] += i_density[p];
            }
        }
        
        if ((t % 1000) == 0) printf(" c = %8d  t = %8d  #e = %8d  #i = %8d\n", cycle,t,N_e,N_i);
    }
    fprintf(datafile,"%8d  %8d  %8d\n",cycle,N_e,N_i);
}

//---------------------------------------------------------------------//
// save particle coordinates                                           //
//---------------------------------------------------------------------//

void save_particle_data(){
    double   d;
    FILE   * f;
    char fname[80];
    
    strcpy(fname,"picdata.bin");
    f = fopen(fname,"wb");
    fwrite(&Time,sizeof(double),1,f);
    d = (double)(cycles_done);
    fwrite(&d,sizeof(double),1,f);
    d = (double)(N_e);
    fwrite(&d,sizeof(double),1,f);
    fwrite(x_e, sizeof(double),N_e,f);
    fwrite(vx_e,sizeof(double),N_e,f);
    fwrite(vy_e,sizeof(double),N_e,f);
    fwrite(vz_e,sizeof(double),N_e,f);
    d = (double)(N_i);
    fwrite(&d,sizeof(double),1,f);
    fwrite(x_i, sizeof(double),N_i,f);
    fwrite(vx_i,sizeof(double),N_i,f);
    fwrite(vy_i,sizeof(double),N_i,f);
    fwrite(vz_i,sizeof(double),N_i,f);
    fclose(f);
    printf(">> eduPIC: data saved : %d electrons %d ions, %d cycles completed, time is %e [s]\n",N_e,N_i,cycles_done,Time);
}

//---------------------------------------------------------------------//
// load particle coordinates                                           //
//---------------------------------------------------------------------//

void load_particle_data(){
    double   d;
    FILE   * f;
    char fname[80];
    
    strcpy(fname,"picdata.bin");
    f = fopen(fname,"rb");
    if (f==NULL) {printf(">> eduPIC: ERROR: No particle data file found, try running initial cycle using argument '0'\n"); exit(0); }
    fread(&Time,sizeof(double),1,f);
    fread(&d,sizeof(double),1,f);
    cycles_done = int(d);
    fread(&d,sizeof(double),1,f);
    N_e = int(d);
    fread(x_e, sizeof(double),N_e,f);
    fread(vx_e,sizeof(double),N_e,f);
    fread(vy_e,sizeof(double),N_e,f);
    fread(vz_e,sizeof(double),N_e,f);
    fread(&d,sizeof(double),1,f);
    N_i = int(d);
    fread(x_i, sizeof(double),N_i,f);
    fread(vx_i,sizeof(double),N_i,f);
    fread(vy_i,sizeof(double),N_i,f);
    fread(vz_i,sizeof(double),N_i,f);
    fclose(f);
    printf(">> eduPIC: data loaded : %d electrons %d ions, %d cycles completed before, time is %e [s]\n",N_e,N_i,cycles_done,Time);
}

//---------------------------------------------------------------------//
// save density data                                                   //
//---------------------------------------------------------------------//

void save_density(void){
    FILE *f;
    double c;
    int m;
    
    f = fopen("density.dat","w");
    c = 1.0 / (double)(no_of_cycles) / (double)(N_T);
    for(m=0; m<N_G; m++){
        fprintf(f,"%8.5f  %12e  %12e\n",m * DX, cumul_e_density[m] * c, cumul_i_density[m] * c);
    }
    fclose(f);
}

//---------------------------------------------------------------------//
// save EEPF data                                                      //
//---------------------------------------------------------------------//

void save_eepf(void) {
    FILE   *f;
    int    i;
    double h,energy;
    
    h = 0.0;
    for (i=0; i<N_EEPF; i++) {h += eepf[i];}
    h *= DE_EEPF;
    f = fopen("eepf.dat","w");
    for (i=0; i<N_EEPF; i++) {
        energy = (i + 0.5) * DE_EEPF;
        fprintf(f,"%e  %e\n", energy, eepf[i] / h / sqrt(energy));
    }
    fclose(f);
}

//---------------------------------------------------------------------//
// save IFED data                                                      //
//---------------------------------------------------------------------//

void save_ifed(void) {
    FILE   *f;
    int    i;
    double h_pow,h_gnd,energy;
    
    h_pow = 0.0;
    h_gnd = 0.0;
    for (i=0; i<N_IFED; i++) {h_pow += ifed_pow[i]; h_gnd += ifed_gnd[i];}
    h_pow *= DE_IFED;
    h_gnd *= DE_IFED;
    mean_i_energy_pow = 0.0;
    mean_i_energy_gnd = 0.0;
    f = fopen("ifed.dat","w");
    for (i=0; i<N_IFED; i++) {
        energy = (i + 0.5) * DE_IFED;
        fprintf(f,"%6.2f %10.6f %10.6f\n", energy, (double)(ifed_pow[i])/h_pow, (double)(ifed_gnd[i])/h_gnd);
        mean_i_energy_pow += energy * (double)(ifed_pow[i]) / h_pow;
        mean_i_energy_gnd += energy * (double)(ifed_gnd[i]) / h_gnd;
    }
    fclose(f);
}

//--------------------------------------------------------------------//
// save XT data                                                       //
//--------------------------------------------------------------------//

void save_xt_1(xt_distr distr, char *fname) {
    FILE   *f;
    int    i, j;
    
    f = fopen(fname,"w");
    for (i=0; i<N_G; i++){
        for (j=0; j<N_XT; j++){
            fprintf(f,"%e  ", distr[i][j]);
        }
        fprintf(f,"\n");
    }
    fclose(f);
}

void norm_all_xt(void){
    double f1, f2;
    int    i, j;
    
    // normalize all XT data
    
    f1 = (double)(N_XT) / (double)(no_of_cycles * N_T);
    f2 = WEIGHT / (ELECTRODE_AREA * DX) / (no_of_cycles * (PERIOD / (double)(N_XT)));
    
    for (i=0; i<N_G; i++){
        for (j=0; j<N_XT; j++){
            pot_xt[i][j]    *= f1;
            efield_xt[i][j] *= f1;
            ne_xt[i][j]     *= f1;
            ni_xt[i][j]     *= f1;
            if (counter_e_xt[i][j] > 0) {
                ue_xt[i][j]     =  ue_xt[i][j] / counter_e_xt[i][j];
                je_xt[i][j]     = -ue_xt[i][j] * ne_xt[i][j] * E_CHARGE;
                meanee_xt[i][j] =  meanee_xt[i][j] / counter_e_xt[i][j];
                ioniz_rate_xt[i][j] *= f2;
             } else {
                ue_xt[i][j]         = 0.0;
                je_xt[i][j]         = 0.0;
                meanee_xt[i][j]     = 0.0;
                ioniz_rate_xt[i][j] = 0.0;
            }
            if (counter_i_xt[i][j] > 0) {
                ui_xt[i][j]     = ui_xt[i][j] / counter_i_xt[i][j];
                ji_xt[i][j]     = ui_xt[i][j] * ni_xt[i][j] * E_CHARGE;
                meanei_xt[i][j] = meanei_xt[i][j] / counter_i_xt[i][j];
            } else {
                ui_xt[i][j]     = 0.0;
                ji_xt[i][j]     = 0.0;
                meanei_xt[i][j] = 0.0;
            }
            powere_xt[i][j] = je_xt[i][j] * efield_xt[i][j];
            poweri_xt[i][j] = ji_xt[i][j] * efield_xt[i][j];
        }
    }
}

void save_all_xt(void){
    char fname[80];
    
    strcpy(fname,"pot_xt.dat");     save_xt_1(pot_xt, fname);
    strcpy(fname,"efield_xt.dat");  save_xt_1(efield_xt, fname);
    strcpy(fname,"ne_xt.dat");      save_xt_1(ne_xt, fname);
    strcpy(fname,"ni_xt.dat");      save_xt_1(ni_xt, fname);
    strcpy(fname,"je_xt.dat");      save_xt_1(je_xt, fname);
    strcpy(fname,"ji_xt.dat");      save_xt_1(ji_xt, fname);
    strcpy(fname,"powere_xt.dat");  save_xt_1(powere_xt, fname);
    strcpy(fname,"poweri_xt.dat");  save_xt_1(poweri_xt, fname);
    strcpy(fname,"meanee_xt.dat");  save_xt_1(meanee_xt, fname);
    strcpy(fname,"meanei_xt.dat");  save_xt_1(meanei_xt, fname);
    strcpy(fname,"ioniz_xt.dat");   save_xt_1(ioniz_rate_xt, fname);
}

//---------------------------------------------------------------------//
// simulation report including stability and accuracy conditions       //
//---------------------------------------------------------------------//

void check_and_save_info(void){
    FILE     *f;
    double   plas_freq, meane, kT, debye_length, density, ecoll_freq, icoll_freq, sim_time, e_max, v_max, power_e, power_i, c;
    int      i,j;
    bool     conditions_OK;
    
    density    = cumul_e_density[N_G / 2] / (double)(no_of_cycles) / (double)(N_T);  // e density @ center
    plas_freq  = E_CHARGE * sqrt(density / EPSILON0 / E_MASS);                       // e plasma frequency @ center
    meane      = mean_energy_accu_center / (double)(mean_energy_counter_center);     // e mean energy @ center
    kT         = 2.0 * meane * EV_TO_J / 3.0;                                        // k T_e @ center (approximate)
    sim_time   = (double)(no_of_cycles) / FREQUENCY;                                 // simulated time
    ecoll_freq = (double)(N_e_coll) / sim_time / (double)(N_e);                      // e collision frequency
    icoll_freq = (double)(N_i_coll) / sim_time / (double)(N_i);                      // ion collision frequency
    debye_length = sqrt(EPSILON0 * kT / density) / E_CHARGE;                         // e Debye length @ center
    
    f = fopen("info.txt","w");
    fprintf(f,"########################## eduPIC simulation report ############################\n");
    fprintf(f,"Simulation parameters:\n");
    fprintf(f,"Gap distance                          = %12.3e [m]\n",  L);
    fprintf(f,"# of grid divisions                   = %12d\n",      N_G);
    fprintf(f,"Frequency                             = %12.3e [Hz]\n", FREQUENCY);
    fprintf(f,"# of time steps / period              = %12d\n",      N_T);
    fprintf(f,"# of electron / ion time steps        = %12d\n",      N_SUB);
    fprintf(f,"Voltage amplitude                     = %12.3e [V]\n",  VOLTAGE);
    fprintf(f,"Pressure (Ar)                         = %12.3e [Pa]\n", PRESSURE);
    fprintf(f,"Temperature                           = %12.3e [K]\n",  TEMPERATURE);
    fprintf(f,"Superparticle weight                  = %12.3e\n",      WEIGHT);
    fprintf(f,"# of simulation cycles in this run    = %12d\n",      no_of_cycles);
    fprintf(f,"--------------------------------------------------------------------------------\n");
    fprintf(f,"Plasma characteristics:\n");
    fprintf(f,"Electron density @ center             = %12.3e [m^{-3}]\n", density);
    fprintf(f,"Plasma frequency @ center             = %12.3e [rad/s]\n",  plas_freq);
    fprintf(f,"Debye length @ center                 = %12.3e [m]\n",      debye_length);
    fprintf(f,"Electron collision frequency          = %12.3e [1/s]\n",    ecoll_freq);
    fprintf(f,"Ion collision frequency               = %12.3e [1/s]\n",    icoll_freq);
    fprintf(f,"--------------------------------------------------------------------------------\n");
    fprintf(f,"Stability and accuracy conditions:\n");
    conditions_OK = true;
    c = plas_freq * DT_E;
    fprintf(f,"Plasma frequency @ center * DT_E      = %12.3f (OK if less than 0.20)\n", c);
    if (c > 0.2) {conditions_OK = false;}
    c = DX / debye_length;
    fprintf(f,"DX / Debye length @ center            = %12.3f (OK if less than 1.00)\n", c);
    if (c > 1.0) {conditions_OK = false;}
    c = max_electron_coll_freq() * DT_E;
    fprintf(f,"Max. electron coll. frequency * DT_E  = %12.3f (OK if less than 0.05)\n", c);
    if (c > 0.05) {conditions_OK = false;}
    c = max_ion_coll_freq() * DT_I;
    fprintf(f,"Max. ion coll. frequency * DT_I       = %12.3f (OK if less than 0.05)\n", c);
    if (c > 0.05) {conditions_OK = false;}
    if (conditions_OK == false){
        fprintf(f,"--------------------------------------------------------------------------------\n");
        fprintf(f,"** STABILITY AND ACCURACY CONDITION(S) VIOLATED - REFINE SIMULATION SETTINGS! **\n");
        fprintf(f,"--------------------------------------------------------------------------------\n");
        fclose(f);
        printf(">> eduPIC: ERROR: STABILITY AND ACCURACY CONDITION(S) VIOLATED!\n");
        printf(">> eduPIC: for details see 'info.txt' and refine simulation settings!\n");
    }
    else
    {
        // calculate maximum energy for which the Courant-Friedrichs-Levy condition holds:
        
        v_max = DX / DT_E;
        e_max = 0.5 * E_MASS * v_max * v_max / EV_TO_J;
        fprintf(f,"Max e- energy for CFL condition       = %12.3f [eV]\n", e_max);
        fprintf(f,"Check EEPF to ensure that CFL is fulfilled for the majority of the electrons!\n");
        fprintf(f,"--------------------------------------------------------------------------------\n");
        
        // saving of the following data is done here as some of the further lines need data
        // that are computed / normalized in these functions
        
        printf(">> eduPIC: saving diagnostics data\n");
        save_density();
        save_eepf();
        save_ifed();
        norm_all_xt();
        save_all_xt();
        fprintf(f,"Particle characteristics at the electrodes:\n");
        fprintf(f,"Ion flux at powered electrode         = %12.3e [m^{-2} s^{-1}]\n", N_i_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
        fprintf(f,"Ion flux at grounded electrode        = %12.3e [m^{-2} s^{-1}]\n", N_i_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
        fprintf(f,"Mean ion energy at powered electrode  = %12.3e [eV]\n", mean_i_energy_pow);
        fprintf(f,"Mean ion energy at grounded electrode = %12.3e [eV]\n", mean_i_energy_gnd);
        fprintf(f,"Electron flux at powered electrode    = %12.3e [m^{-2} s^{-1}]\n", N_e_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
        fprintf(f,"Electron flux at grounded electrode   = %12.3e [m^{-2} s^{-1}]\n", N_e_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD));
        fprintf(f,"--------------------------------------------------------------------------------\n");
        
        // calculate spatially and temporally averaged power absorption by the electrons and ions
        
        power_e = 0.0;
        power_i = 0.0;
        for (i=0; i<N_G; i++){
            for (j=0; j<N_XT; j++){
                power_e += powere_xt[i][j];
                power_i += poweri_xt[i][j];
            }
        }
        power_e /= (double)(N_XT * N_G);
        power_i /= (double)(N_XT * N_G);
        fprintf(f,"Absorbed power calculated as <j*E>:\n");
        fprintf(f,"Electron power density (average)      = %12.3e [W m^{-3}]\n", power_e);
        fprintf(f,"Ion power density (average)           = %12.3e [W m^{-3}]\n", power_i);
        fprintf(f,"Total power density(average)          = %12.3e [W m^{-3}]\n", power_e + power_i);
        fprintf(f,"--------------------------------------------------------------------------------\n");
        fclose(f);
    }
}

//------------------------------------------------------------------------------------------//
// main                                                                                     //
// command line arguments:                                                                  //
// [1]: number of cycles (0 for init)                                                       //
// [2]: "m" turns on data collection and saving                                             //
//------------------------------------------------------------------------------------------//

int main (int argc, char *argv[]){
    printf(">> eduPIC: starting...\n");
    printf(">> eduPIC: **************************************************************************\n");
    printf(">> eduPIC: Copyright (C) 2021 Z. Donko et al.\n");
    printf(">> eduPIC: This program comes with ABSOLUTELY NO WARRANTY\n");
    printf(">> eduPIC: This is free software, you are welcome to use, modify and redistribute it\n");
    printf(">> eduPIC: according to the GNU General Public License, https://www.gnu.org/licenses/\n");
    printf(">> eduPIC: **************************************************************************\n");

    if (argc == 1) {
        printf(">> eduPIC: error = need starting_cycle argument\n");
        return 1;
    } else {
        strcpy(st0,argv[1]);
        arg1 = atol(st0);
        if (argc > 2) {
            if (strcmp (argv[2],"m") == 0){
                measurement_mode = true;                  // measurements will be done
            } else {
                measurement_mode = false;
            }
        }
    }
    if (measurement_mode) {
        printf(">> eduPIC: measurement mode: on\n");
    } else {
        printf(">> eduPIC: measurement mode: off\n");
    }
    set_electron_cross_sections_ar();
    set_ion_cross_sections_ar();
    calc_total_cross_sections();
    //test_cross_sections(); return 1;
    datafile = fopen("conv.dat","a");
    if (arg1 == 0) {
        if (FILE *file = fopen("picdata.bin", "r")) { fclose(file);
            printf(">> eduPIC: Warning: Data from previous calculation are detected.\n");
            printf("           To start a new simulation from the beginning, please delete all output files before running ./eduPIC 0\n");
            printf("           To continue the existing calculation, please specify the number of cycles to run, e.g. ./eduPIC 100\n");
            exit(0);
        } 
        no_of_cycles = 1;
        cycle = 1;                                        // init cycle
        init(N_INIT);                                     // seed initial electrons & ions
        printf(">> eduPIC: running initializing cycle\n");
        Time = 0;
        do_one_cycle();
        cycles_done = 1;
    } else {
        no_of_cycles = arg1;                              // run number of cycles specified in command line
        load_particle_data();                             // read previous configuration from file
        printf(">> eduPIC: running %d cycle(s)\n",no_of_cycles);
        for (cycle=cycles_done+1;cycle<=cycles_done+no_of_cycles;cycle++) {do_one_cycle();}
        cycles_done += no_of_cycles;
    }
    fclose(datafile);
    save_particle_data();
    if (measurement_mode) {
        check_and_save_info();
    }
    printf(">> eduPIC: simulation of %d cycle(s) is completed.\n",no_of_cycles);
}
