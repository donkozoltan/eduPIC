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
// Plasma Sources Science and Technology, vol 30, pp. 095017 (2021)  //
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



#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <array>
#include <algorithm>
#include <numeric>
#include <string>

using namespace std;

// constants

const double     PI             = 3.141592653589793;      // mathematical constant Pi
const double     TWO_PI         = 2.0 * PI;               // two times Pi
const double     E_CHARGE       = 1.60217662e-19;         // electron charge [C]
const double     EV_TO_J        = E_CHARGE;               // eV <-> Joule conversion factor
const double     E_MASS         = 9.10938356e-31;         // mass of electron [kg]
const double     AR_MASS        = 6.63352090e-26;         // mass of argon atom [kg]
const double     MU_ARAR        = AR_MASS / 2.0;          // reduced mass of two argon atoms [kg]
const double     K_BOLTZMANN    = 1.38064852e-23;         // Boltzmann's constant [J/K]
const double     EPSILON0       = 8.85418781e-12;         // permittivity of free space [F/m]

// simulation parameters

const int        N_G            = 400;                    // number of grid points
const int        N_T            = 4000;                   // time steps within an RF period
const double     FREQUENCY      = 13.56e6;                // driving frequency [Hz]
const double     VOLTAGE        = 250.0;                  // voltage amplitude [V]
const double     L              = 0.025;                  // electrode gap [m]
const double     PRESSURE       = 10.0;                   // gas pressure [Pa]
const double     TEMPERATURE    = 350.0;                  // background gas temperature [K]
const double     WEIGHT         = 7.0e4;                  // weight of superparticles
const double     ELECTRODE_AREA = 1.0e-4;                 // (fictive) electrode area [m^2]
const int        N_INIT         = 1000;                   // number of initial electrons and ions

// additional (derived) constants

const double     PERIOD         = 1.0 / FREQUENCY;                           // RF period length [s]
const double     DT_E           = PERIOD / static_cast<double>(N_T);         // electron time step [s]
const int        N_SUB          = 20;                                        // ions move only in these cycles (subcycling)
const double     DT_I           = N_SUB * DT_E;                              // ion time step [s]
const double     DX             = L / static_cast<double>(N_G - 1);          // spatial grid division [m]
const double     INV_DX         = 1.0 / DX;                                  // inverse of spatial grid size [1/m]
const double     GAS_DENSITY    = PRESSURE / (K_BOLTZMANN * TEMPERATURE);    // background gas density [1/m^3]
const double     OMEGA          = TWO_PI * FREQUENCY;                        // angular frequency [rad/s]

// electron and ion cross sections

const int        N_CS           = 5;                      // total number of processes / cross sections
const int        E_ELA          = 0;                      // process identifier: electron/elastic
const int        E_EXC          = 1;                      // process identifier: electron/excitation
const int        E_ION          = 2;                      // process identifier: electron/ionization
const int        I_ISO          = 3;                      // process identifier: ion/elastic/isotropic
const int        I_BACK         = 4;                      // process identifier: ion/elastic/backscattering
const double     E_EXC_TH       = 11.5;                   // electron impact excitation threshold [eV]
const double     E_ION_TH       = 15.8;                   // electron impact ionization threshold [eV]
const int        CS_RANGES      = 1000000;                // number of entries in cross section arrays
const double     DE_CS          = 0.001;                  // energy division in cross section arrays [eV]
using cross_section = array<float,CS_RANGES>;             // cross section array
 
cross_section    sigma[N_CS];                             // set of cross section arrays
cross_section    sigma_tot_e;                             // total macroscopic cross section of electrons
cross_section    sigma_tot_i;                             // total macroscopic cross section of ions

// particle coordinates

size_t     N_e = 0;                                       // number of electrons
size_t     N_i = 0;                                       // number of ions

vector<double>  x_e, vx_e, vy_e, vz_e;                    // coordinates of electrons (one spatial, three velocity components)
vector<double>  x_i, vx_i, vy_i, vz_i;                    // coordinates of ions (one spatial, three velocity components)

using xvector = array<double,N_G>;                        // array for quantities defined at gird points
xvector  efield, pot;                                     // electric field and potential
xvector  e_density, i_density;                            // electron and ion densities
xvector  cumul_e_density, cumul_i_density;                // cumulative densities

using Ullong = unsigned long long int;                    // compact name for 64 bit unsigned integer
Ullong   N_e_abs_pow  = 0;                                // counter for electrons absorbed at the powered electrode
Ullong   N_e_abs_gnd  = 0;                                // counter for electrons absorbed at the grounded electrode
Ullong   N_i_abs_pow  = 0;                                // counter for ions absorbed at the powered electrode
Ullong   N_i_abs_gnd  = 0;                                // counter for ions absorbed at the grounded electrode

// electron energy probability function

const int    N_EEPF  = 2000;                              // number of energy bins in Electron Energy Probability Function (EEPF)
const double DE_EEPF = 0.05;                              // resolution of EEPF [eV]
using eepf_vector = array<double,N_EEPF>;                 // array for EEPF
eepf_vector  eepf   {0.0};                                // time integrated EEPF in the center of the plasma

// ion flux-energy distributions

const int    N_IFED  = 200;                               // number of energy bins in Ion Flux-Energy Distributions (IFEDs)
const double DE_IFED = 1.0;                               // resolution of IFEDs [eV]
using ifed_vector = array<int, N_IFED>;                   // array for IFEDs
ifed_vector  ifed_pow {0};                                // IFED at the powered electrode
ifed_vector  ifed_gnd {0};                                // IFED at the grounded electrode
double       mean_i_energy_pow;                           // mean ion energy at the powered electrode
double       mean_i_energy_gnd;                           // mean ion energy at the grounded electrode

// spatio-temporal (XT) distributions

const int N_BIN                      = 20;                // number of time steps binned for the XT distributions
const int N_XT                       = N_T / N_BIN;       // number of spatial bins for the XT distributions
using xt_distr = array<double,N_G*N_XT>;                  // array for XT distributions (decimal numbers)


xt_distr pot_xt                      = {0.0};             // XT distribution of the potential
xt_distr efield_xt                   = {0.0};             // XT distribution of the electric field
xt_distr ne_xt                       = {0.0};             // XT distribution of the electron density
xt_distr ni_xt                       = {0.0};             // XT distribution of the ion density
xt_distr ue_xt                       = {0.0};             // XT distribution of the mean electron velocity
xt_distr ui_xt                       = {0.0};             // XT distribution of the mean ion velocity
xt_distr je_xt                       = {0.0};             // XT distribution of the electron current density
xt_distr ji_xt                       = {0.0};             // XT distribution of the ion current density
xt_distr powere_xt                   = {0.0};             // XT distribution of the electron powering (power absorption) rate
xt_distr poweri_xt                   = {0.0};             // XT distribution of the ion powering (power absorption) rate
xt_distr meanee_xt                   = {0.0};             // XT distribution of the mean electron energy
xt_distr meanei_xt                   = {0.0};             // XT distribution of the mean ion energy
xt_distr counter_e_xt                = {0.0};             // XT counter for electron properties
xt_distr counter_i_xt                = {0.0};             // XT counter for ion properties
xt_distr ioniz_rate_xt               = {0.0};             // XT distribution of the ionisation rate

double    mean_energy_accu_center    = 0;                 // mean electron energy accumulator in the center of the gap
Ullong    mean_energy_counter_center = 0;                 // mean electron energy counter in the center of the gap
Ullong    N_e_coll                   = 0;                 // counter for electron collisions
Ullong    N_i_coll                   = 0;                 // counter for ion collisions

double    Time;                                           // total simulated time (from the beginning of the simulation)
int       cycle,no_of_cycles,cycles_done;                 // current cycle and total cycles in the run, cycles completed (from the beginning of the simulation)

int       arg1;                                           // used for reading command line arguments
bool      measurement_mode;                               // flag that controls measurements and data saving

ofstream  datafile("conv.dat",ios_base::app);             // stream to external file for saving convergence data

//------------------------------------------------------------------------//
// C++ Mersenne Twister 19937 generator                                   //
// R01(MTgen) will genarate uniform distribution over [0,1) interval      //
// RMB(MTgen) will generate Maxwell-Boltzmann distribution (of gas atoms) //
//------------------------------------------------------------------------//

std::random_device rd{}; 
std::mt19937 MTgen(rd());
std::uniform_real_distribution<> R01(0.0, 1.0);
std::normal_distribution<> RMB(0.0,sqrt(K_BOLTZMANN * TEMPERATURE / AR_MASS));

//----------------------------------------------------------------------------//
//  electron cross sections: A V Phelps & Z Lj Petrovic, PSST 8 R21 (1999)    //
//----------------------------------------------------------------------------//

void set_electron_cross_sections_ar(void){
    cout<<">> eduPIC: Setting e- / Ar cross sections"<<endl;

    auto qmel = [](auto en){ return 1e-20*(fabs(6.0 / pow(1.0 + (en/0.1) + pow(en/0.6,2.0), 3.3)
        - 1.1 * pow(en, 1.4) / (1.0 + pow(en/15.0, 1.2)) / sqrt(1.0 + pow(en/5.5, 2.5) + pow(en/60.0, 4.1)))
        + 0.05 / pow(1.0 + en/10.0, 2.0) + 0.01 * pow(en, 3.0) / (1.0 + pow(en/12.0, 6.0))); };

    auto qexc= [](const auto &en){ if(en>E_EXC_TH){ return 1e-20*(0.034 * pow(en-11.5, 1.1) * (1.0 + pow(en/15.0, 2.8)) / (1.0 + pow(en/23.0, 5.5))
            + 0.023 * (en-11.5) / pow(1.0 + en/80.0, 1.9)); } else { return 0.0;} };

    auto qion = [](const auto &en){ if(en>E_ION_TH){ return 1e-20*(970.0 * (en-15.8) / pow(70.0 + en, 2.0) +
                0.06 * pow(en-15.8, 2.0) * exp(-en/9)); } else {return 0.0;} };

    vector<float> e(CS_RANGES);
    e[0]=DE_CS;             
    generate(e.begin()+1,e.end(),[i=1]()mutable{return DE_CS*(i++); });   // electron energy

    transform(e.begin(),e.end(),sigma[E_ELA].begin(),qmel);    // cross section for e- / Ar elastic collision
    transform(e.begin(),e.end(),sigma[E_EXC].begin(),qexc);    // cross section for e- / Ar excitation
    transform(e.begin(),e.end(),sigma[E_ION].begin(),qion);    // cross section for e- / Ar ionization
}

//-------------------------------------------------------------------//
//  ion cross sections: A. V. Phelps, J. Appl. Phys. 76, 747 (1994)  //
//-------------------------------------------------------------------//

void set_ion_cross_sections_ar(void){
    cout<<">> eduPIC: Setting Ar+ / Ar cross sections"<<endl;
    auto qiso = [](const auto &e_lab){ return 2e-19 * pow(e_lab,-0.5) / (1.0 + e_lab) +
                                    3e-19 * e_lab / pow(1.0 + e_lab / 3.0, 2.0); };

    auto qmom= [](const auto &e_lab){ return 1.15e-18 * pow(e_lab,-0.1) * pow(1.0 + 0.015 / e_lab, 0.6); };

    auto qback = [&](const auto &x){ return (qmom(x)-qiso(x))/2.0; };

    vector<float> e(CS_RANGES);
    e[0]=2.0*DE_CS;
    generate(e.begin()+1,e.end(),[i=1]()mutable{return 2.0*DE_CS*(i++); });   // ion energy in the laboratory frame of reference

    transform(e.begin(),e.end(),sigma[I_ISO].begin(),qiso);     // cross section for Ar+ / Ar isotropic part of elastic scattering
    transform(e.begin(),e.end(),sigma[I_BACK].begin(),qback);   // cross section for Ar+ / Ar backward elastic scattering
}

//----------------------------------------------------------------------//
//  calculation of total cross sections for electrons and ions          //
//----------------------------------------------------------------------//

void calc_total_cross_sections(void){

    for(size_t i{0}; i<CS_RANGES; ++i){
        sigma_tot_e[i] = (sigma[E_ELA][i] + sigma[E_EXC][i] + sigma[E_ION][i]) * GAS_DENSITY;   // total macroscopic cross section of electrons
        sigma_tot_i[i] = (sigma[I_ISO][i] + sigma[I_BACK][i]) * GAS_DENSITY;                    // total macroscopic cross section of ions
    }
}

//----------------------------------------------------------------------//
//  test of cross sections for electrons and ions                       //
//----------------------------------------------------------------------//

void test_cross_sections(void){
    ofstream f("cross_sections.dat");                             // cross sections saved in data file: cross_sections.dat
    ostream_iterator<float> tofile(f, "\n");

    for(size_t i{0}; i<CS_RANGES;++i){f<<i*DE_CS<<endl;}
    for(const auto & v:sigma){
        copy(v.begin(),v.end(),tofile);
    }
    f.close();
}

//---------------------------------------------------------------------//
// find upper limit of collision frequencies                           //
//---------------------------------------------------------------------//

double max_electron_coll_freq (void){
    double e,v,nu,nu_max;
    nu_max = 0;
    for(size_t i{0}; i<CS_RANGES; ++i){
        e  = i * DE_CS;
        v  = sqrt(2.0 * e * EV_TO_J / E_MASS);
        nu = v * sigma_tot_e[i];
        if (nu > nu_max) {nu_max = nu;}
    }
    return nu_max;
}

double max_ion_coll_freq (void){
    double e,g,nu,nu_max;
    nu_max = 0;
    for(size_t i{0}; i<CS_RANGES; ++i){
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
    N_e = nseed;
    N_i = nseed;
    x_e.resize(nseed);
    x_i.resize(nseed);
    vx_e.resize(nseed, 0.0);                                      // initial velocity components of the electron
    vy_e.resize(nseed, 0.0);
    vz_e.resize(nseed, 0.0);
    vx_i.resize(nseed, 0.0);                                      // initial velocity components of the ion
    vy_i.resize(nseed, 0.0);
    vz_i.resize(nseed, 0.0);
    generate(x_e.begin(),x_e.end(),[=](){return L*R01(MTgen);});  // initial random position of the electron
    generate(x_i.begin(),x_i.end(),[=](){return L*R01(MTgen);});  // initial random position of the ion
}

//----------------------------------------------------------------------//
// e / Ar collision  (cold gas approximation)                           //
//----------------------------------------------------------------------//

void collision_electron (double xe, double &vxe, double &vye, double &vze, const int &eindex){
    const double F1 = E_MASS  / (E_MASS + AR_MASS);
    const double F2 = AR_MASS / (E_MASS + AR_MASS);
    double t0,t1,t2,rnd;
    double g,g2,gx,gy,gz,wx,wy,wz,theta,phi;
    double chi,eta,chi2,eta2,sc,cc,se,ce,st,ct,sp,cp,energy,e_sc,e_ej;
    
    // calculate relative velocity before collision & velocity of the centre of mass
    
    gx = vxe;                             
    gy = vye;
    gz = vze;
    g  = sqrt(gx * gx + gy * gy + gz * gz);
    wx = F1 * vxe;
    wy = F1 * vye;
    wz = F1 * vze;
    
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
        g = sqrt(2.0 * energy / E_MASS);             // relative velocity after energy loss
        chi = acos(1.0 - 2.0 * R01(MTgen));          // isotropic scattering
        eta = TWO_PI * R01(MTgen);                   // azimuthal angle
    } else {                                         // ionization
        energy = 0.5 * E_MASS * g * g;               // electron energy
        energy = fabs(energy - E_ION_TH * EV_TO_J);  // subtract energy loss for ionization
        e_ej = 10.0 * tan(R01(MTgen) * atan(energy/EV_TO_J / 20.0)) * EV_TO_J;   // energy of the emitted electron
        e_sc = fabs(energy - e_ej);                  // energy of incoming electron after collision
        g    = sqrt(2.0 * e_sc / E_MASS);            // relative velocity of incoming (original) electron
        g2   = sqrt(2.0 * e_ej / E_MASS);            // relative velocity of emitted (new) electron
        chi  = acos(sqrt(e_sc / energy));            // scattering angle for incoming electron
        chi2 = acos(sqrt(e_ej / energy));            // scattering angle for emitted electrons
        eta  = TWO_PI * R01(MTgen);                  // azimuthal angle for incoming electron
        eta2 = eta + PI;                             // azimuthal angle for emitted electron
        sc  = sin(chi2);
        cc  = cos(chi2);
        se  = sin(eta2);
        ce  = cos(eta2);
        gx  = g2 * (ct * cc - st * sc * ce);
        gy  = g2 * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
        gz  = g2 * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
        N_e++;                                        // add new electron
        x_e.push_back(xe);
        vx_e.push_back(wx + F2 * gx);
        vy_e.push_back(wy + F2 * gy);
        vz_e.push_back(wz + F2 * gz);
        N_i++;                                        // add new ion
        x_i.push_back(xe);
        vx_i.push_back(RMB(MTgen));                   // velocity is sampled from background thermal distribution
        vy_i.push_back(RMB(MTgen));
        vz_i.push_back(RMB(MTgen));
    }
    
    // scatter the primary electron

    sc  = sin(chi);
    cc  = cos(chi);
    se  = sin(eta);
    ce  = cos(eta);
    
    // compute new relative velocity:
    
    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
    
    // post-collision velocity of the colliding electron
    
    vxe = wx + F2 * gx;
    vye = wy + F2 * gy;
    vze = wz + F2 * gz;
}

//----------------------------------------------------------------------//
// Ar+ / Ar collision                                                   //
//----------------------------------------------------------------------//

void collision_ion (double &vx_1, double &vy_1, double &vz_1,
                    double &vx_2, double &vy_2, double &vz_2, const int &e_index){
    double   g,gx,gy,gz,wx,wy,wz,rnd;
    double   theta,phi,chi,eta,st,ct,sp,cp,sc,cc,se,ce,t1,t2;
    
    // calculate relative velocity before collision
    // random Maxwellian target atom already selected (vx_2,vy_2,vz_2 velocity components of target atom come with the call)
    

    gx = vx_1-vx_2;
    gy = vy_1-vy_2;
    gz = vz_1-vz_2;
    g  = sqrt(gx * gx + gy * gy + gz * gz);
    wx = 0.5 * (vx_1 + vx_2);
    wy = 0.5 * (vy_1 + vy_2);
    wz = 0.5 * (vz_1 + vz_2);

    // find Euler angles:

    if (gx == 0) {theta = 0.5 * PI;} else {theta = atan2(sqrt(gy * gy + gz * gz),gx);}
    if (gy == 0) {
        if (gz > 0){phi = 0.5 * PI;} else {phi = - 0.5 * PI;}
    } else {phi = atan2(gz, gy);}


    // determine the type of collision based on cross sections and generate scattering angle

    t1  =     sigma[I_ISO][e_index];
    t2  = t1 +sigma[I_BACK][e_index];
    rnd = R01(MTgen);
    if  (rnd < (t1 /t2)){                                  // isotropic scattering
        chi = acos(1.0 - 2.0 * R01(MTgen));                // isotropic scattering angle
    } else {                                               // backward scattering
        chi = PI;                                          // backward scattering angle
    }
    eta = TWO_PI * R01(MTgen);                             // azimuthal angle
    sc  = sin(chi);
    cc  = cos(chi);
    se  = sin(eta);
    ce  = cos(eta);
    st  = sin(theta);
    ct  = cos(theta);
    sp  = sin(phi);
    cp  = cos(phi);

    // compute new relative velocity:

    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);

    // post-collision velocity of the ion

    vx_1 = wx + 0.5 * gx;
    vy_1 = wy + 0.5 * gy;
    vz_1 = wz + 0.5 * gz;
}

//----------------------------------------------------------------------
// solve Poisson equation (Thomas algorithm)    
//----------------------------------------------------------------------

void solve_Poisson (const xvector &rho1, const double &tt){
    const double A =  1.0;
    const double B = -2.0;
    const double C =  1.0;
    const double S = 1.0 / (2.0 * DX);
    const double ALPHA = -DX * DX / EPSILON0;
    xvector g, w, f;
    size_t  i;

    // apply potential to the electrodes - boundary conditions

    pot.front() = VOLTAGE * cos(OMEGA * tt);    // potential at the powered electrode
    pot.back()  = 0.0;                          // potential at the grounded electrode

    // solve Poisson equation

    for(i=1; i<=N_G-2; ++i) f[i] = ALPHA * rho1[i];
    f[1] -= pot.front();
    f[N_G-2] -= pot.back();
    w[1] = C/B;
    g[1] = f[1]/B;
    for(i=2; i<=N_G-2; ++i){
        w[i] = C / (B - A * w[i-1]);
        g[i] = (f[i] - A * g[i-1]) / (B - A * w[i-1]);
    }
    pot[N_G-2] = g[N_G-2];
    for (i=N_G-3; i>0; --i) pot[i] = g[i] - w[i] * pot[i+1];    // potential at the grid points between the electrodes

    // compute electric field

    for(i=1; i<=N_G-2; ++i) efield[i] = (pot[i-1] - pot[i+1]) * S;   // electric field at the grid points between the electrodes
    efield.front() = (pot[0]     - pot[1])     * INV_DX - rho1.front() * DX / (2.0 * EPSILON0);   // powered electrode
    efield.back()  = (pot[N_G-2] - pot[N_G-1]) * INV_DX + rho1.back()  * DX / (2.0 * EPSILON0);   // grounded electrode
}

//---------------------------------------------------------------------//
// simulation of one radiofrequency cycle                              //
//---------------------------------------------------------------------//

void do_one_cycle (void){
    const double DV       = ELECTRODE_AREA * DX;
    const double FACTOR_W = WEIGHT / DV;
    const double FACTOR_E = DT_E / E_MASS * E_CHARGE;
    const double FACTOR_I = DT_I / AR_MASS * E_CHARGE;
    const double MIN_X = 0.45 * L;                       // min. position for EEPF collection
    const double MAX_X = 0.55 * L;                       // max. position for EEPF collection
    size_t      k, t, p, energy_index;
    double   rmod, rint, g, g_sqr, gx, gy, gz, vx_a, vy_a, vz_a, e_x, energy, nu, p_coll, v_sqr, velocity;
    double   mean_v, rate;
    bool     out;
    xvector  rho;
    size_t      t_index;

    for (t=0; t<N_T; t++){         // a RF period is divided into N_T equal time intervals (time step DT_E)
        Time += DT_E;              // update of the total simulated time
        t_index = t / N_BIN;       // index for XT distributions

        // step 1: compute densities at grid points

        fill(e_density.begin(),e_density.end(),0.0);             // electron density - computed in every time step
        for(k=0; k<N_e; ++k){
            rmod = modf(x_e[k] * INV_DX, &rint);
            p    = static_cast<int>(rint);
            e_density[p]   += (1.0-rmod) * FACTOR_W;             
            e_density[p+1] += rmod * FACTOR_W;
        }
        e_density.front() *= 2.0;
        e_density.back()  *= 2.0;
        transform(cumul_e_density.begin(),cumul_e_density.end(),e_density.begin(),cumul_e_density.begin(),[](auto x, auto y){return x+y;});

        if ((t % N_SUB) == 0) {                                  // ion density - computed in every N_SUB-th time steps (subcycling)
            fill(i_density.begin(),i_density.end(),0.0);
            for(k=0; k<N_i; ++k){
                rmod = modf(x_i[k] * INV_DX, &rint);
                p    = static_cast<int>(rint);
                i_density[p]   += (1.0-rmod) * FACTOR_W;         
                i_density[p+1] += rmod * FACTOR_W;
            }
            i_density.front() *= 2.0;
            i_density.back()  *= 2.0;
        }
        transform(cumul_i_density.begin(),cumul_i_density.end(),i_density.begin(),cumul_i_density.begin(),[](auto x, auto y){return x+y;});

        // step 2: solve Poisson equation
        
        // get charge density
        transform(i_density.begin(),i_density.end(),e_density.begin(),rho.begin(),[](auto x, auto y){return E_CHARGE*(x-y);});
        solve_Poisson(rho,Time);                                               // compute potential and electric field

        // steps 3 & 4: move particles according to electric field interpolated to particle positions

        for(k=0; k<N_e; k++){                      // move all electrons in every time step
            rmod = modf(x_e[k] * INV_DX, &rint);
            p    = static_cast<int>(rint);
            e_x  = (1.0-rmod)*efield[p] + rmod*efield[p+1];

            if (measurement_mode) {
               
                // measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'
                
                mean_v = vx_e[k] - 0.5 * e_x * FACTOR_E;
                counter_e_xt[p*N_XT+t_index]   += (1.0-rmod);
                counter_e_xt[(p+1)*N_XT+t_index] += rmod;
                ue_xt[p*N_XT+t_index]   += (1.0-rmod) * mean_v;
                ue_xt[(p+1)*N_XT+t_index] += rmod * mean_v;
                v_sqr  = mean_v * mean_v + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k];
                energy = 0.5 * E_MASS * v_sqr / EV_TO_J;
                meanee_xt[p*N_XT+t_index]   += (1.0-rmod) * energy;
                meanee_xt[(p+1)*N_XT+t_index] += rmod * energy;
                energy_index = min( static_cast<int>(energy / DE_CS + 0.5), CS_RANGES-1);
                velocity = sqrt(v_sqr);
                rate = sigma[E_ION][energy_index] * velocity * DT_E * GAS_DENSITY;
                ioniz_rate_xt[p*N_XT+t_index]   += (1.0-rmod) * rate;
                ioniz_rate_xt[(p+1)*N_XT+t_index] += rmod * rate;

                // measure EEPF in the center
                
                if ((MIN_X < x_e[k]) && (x_e[k] < MAX_X)){
                    energy_index = static_cast<int>(energy / DE_EEPF);
                    if (energy_index < N_EEPF) {eepf[energy_index] += 1.0;}
                    mean_energy_accu_center += energy;
                    mean_energy_counter_center++;
                }
            }

            // update velocity and position

            vx_e[k] -= e_x * FACTOR_E;
            x_e[k]  += vx_e[k] * DT_E;
        }

        if ((t % N_SUB) == 0) {                    // move all ions in every N_SUB-th time steps (subcycling)
            for(k=0; k<N_i; k++){
                rmod = modf(x_i[k] * INV_DX, &rint);
                p    = static_cast<int>(rint);
                e_x  = (1.0-rmod)*efield[p] + rmod*efield[p+1];

                if (measurement_mode) {
                    
                    // measurements: 'x' and 'v' are needed at the same time, i.e. old 'x' and mean 'v'

                    mean_v = vx_i[k] + 0.5 * e_x * FACTOR_I;
                    counter_i_xt[p*N_XT+t_index]   += (1.0-rmod);
                    counter_i_xt[(p+1)*N_XT+t_index] += rmod;
                    ui_xt[p*N_XT+t_index]   += (1.0-rmod) * mean_v;
                    ui_xt[(p+1)*N_XT+t_index] += rmod * mean_v;
                    v_sqr  = mean_v * mean_v + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    meanei_xt[p*N_XT+t_index]   += (1.0-rmod) * energy;
                    meanei_xt[(p+1)*N_XT+t_index] += rmod * energy;
                }
                
                // update velocity and position
                
                vx_i[k] += e_x * FACTOR_I;
                x_i[k]  += vx_i[k] * DT_I;
            }
        }
        

        // step 5: check boundaries
        k = 0;
        while(k < x_e.size()) {    // check boundaries for all electrons in every time step
            out = false;
            if (x_e[k] < 0) {N_e_abs_pow++; out = true;}      // the electron is out at the powered electrode
            if (x_e[k] > L) {N_e_abs_gnd++; out = true;}      // the electron is out at the grounded electrode
            if (out) {                                        // remove the electron, if out
                x_e[k]=x_e.back(); x_e.pop_back();
                vx_e[k]=vx_e.back(); vx_e.pop_back();
                vy_e[k]=vy_e.back(); vy_e.pop_back();
                vz_e[k]=vz_e.back(); vz_e.pop_back();
                N_e--;
            } else k++;
        }

        if ((t % N_SUB) == 0) {           // check boundaries for all ions in every N_SUB-th time steps (subcycling)
            k = 0;
            while(k < x_i.size()) {
                out = false;
                if (x_i[k] < 0) {             // the ion is out at the powered electrode
                    N_i_abs_pow++;
                    out    = true;
                    v_sqr  = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    energy_index = static_cast<int>(energy / DE_IFED);
                    if (energy_index < N_IFED) {ifed_pow[energy_index]++;}   // save IFED at the powered electrode
                }
                if (x_i[k] > L) {             // the ion is out at the grounded electrode
                    N_i_abs_gnd++;
                    out    = true;
                    v_sqr  = vx_i[k] * vx_i[k] + vy_i[k] * vy_i[k] + vz_i[k] * vz_i[k];
                    energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                    energy_index = static_cast<int>(energy / DE_IFED);
                    if (energy_index < N_IFED) {ifed_gnd[energy_index]++;}   // save IFED at the grounded electrode
                }
                if (out) {                    // delete the ion, if out
                        x_i[k]=x_i.back(); x_i.pop_back();
                        vx_i[k]=vx_i.back(); vx_i.pop_back();
                        vy_i[k]=vy_i.back(); vy_i.pop_back();
                        vz_i[k]=vz_i.back(); vz_i.pop_back();
                        N_i--;
                } else k++;
            }
        }

        // step 6: collisions

        for (k=0; k<N_e; ++k){                         // checking for occurrence of a collision for all electrons in every time step
            v_sqr = vx_e[k] * vx_e[k] + vy_e[k] * vy_e[k] + vz_e[k] * vz_e[k];
            velocity = sqrt(v_sqr);
            energy   = 0.5 * E_MASS * v_sqr / EV_TO_J;
            energy_index = min(static_cast<int>(energy / DE_CS + 0.5), CS_RANGES-1);
            nu = sigma_tot_e[energy_index] * velocity;
            p_coll = 1 - exp(- nu * DT_E);             // collision probability for electrons
            if (R01(MTgen) < p_coll) {                 // electron collision takes place
                collision_electron(x_e[k], vx_e[k], vy_e[k], vz_e[k], energy_index);
                N_e_coll++;
            }
        }

        if ((t % N_SUB) == 0) {                        // checking for occurrence of a collision for all ions in every N_SUB-th time steps (subcycling)
            for (k=0; k<N_i; ++k){
                vx_a = RMB(MTgen);                     // pick velocity components of a random gas atoms
                vy_a = RMB(MTgen);
                vz_a = RMB(MTgen);
                gx   = vx_i[k] - vx_a;                  // compute the relative velocity of the collision partners
                gy   = vy_i[k] - vy_a;
                gz   = vz_i[k] - vz_a;
                g_sqr = gx * gx + gy * gy + gz * gz;
                g = sqrt(g_sqr);
                energy = 0.5 * MU_ARAR * g_sqr / EV_TO_J;
                energy_index = min( static_cast<int>(energy / DE_CS + 0.5), CS_RANGES-1);    
                nu = sigma_tot_i[energy_index] * g;
                p_coll = 1 - exp(- nu * DT_I);         // collision probability for ions
                if (R01(MTgen)< p_coll) {              // ion collision takes place
                    collision_ion (vx_i[k], vy_i[k], vz_i[k], vx_a, vy_a, vz_a, energy_index);
                    N_i_coll++;
                }
            }
        }

        if (measurement_mode) {

            // collect data from the grid:

            for (p=0; p<N_G; p++) {
                pot_xt   [p*N_XT+t_index] += pot[p];
                efield_xt[p*N_XT+t_index] += efield[p];
                ne_xt    [p*N_XT+t_index] += e_density[p];
                ni_xt    [p*N_XT+t_index] += i_density[p];
            }
        }

           if ((t % 1000) == 0) cout<<" c = "<<setw(8)<<cycle<<"  t = "<<setw(8)<<t<<"  #e = "<<setw(8)<<N_e<<"  #i = "<<setw(8)<<N_i<<endl;
    }
    datafile<<cycle<<" "<<N_e<<" "<<N_i<<endl;
}

//---------------------------------------------------------------------//
// save and load particle coordinates                                  //
//---------------------------------------------------------------------//

void save_particle_data(){
    ofstream f("picdata.bin",ios::binary);

    f.write(reinterpret_cast<char*>(&Time),sizeof(double));
    f.write(reinterpret_cast<char*>(&cycles_done),sizeof(int));
    f.write(reinterpret_cast<char*>(&N_e),sizeof(int));
    f.write(reinterpret_cast<char*>(&x_e[0]),N_e*sizeof(double));
    f.write(reinterpret_cast<char*>(&vx_e[0]),N_e*sizeof(double));
    f.write(reinterpret_cast<char*>(&vy_e[0]),N_e*sizeof(double));
    f.write(reinterpret_cast<char*>(&vz_e[0]),N_e*sizeof(double));
    f.write(reinterpret_cast<char*>(&N_i),sizeof(int));
    f.write(reinterpret_cast<char*>(&x_i[0]),N_i*sizeof(double));
    f.write(reinterpret_cast<char*>(&vx_i[0]),N_i*sizeof(double));
    f.write(reinterpret_cast<char*>(&vy_i[0]),N_i*sizeof(double));
    f.write(reinterpret_cast<char*>(&vz_i[0]),N_i*sizeof(double));
    f.close();

    cout<<">> eduPIC: data saved : "<<N_e<<" electrons "<<N_i<<" ions, "
    <<cycles_done<<" cycles completed, time is "<<scientific<<Time<<" [s]"<<endl;
}

//---------------------------------------------------------------------//
// load particle coordinates                                           //
//---------------------------------------------------------------------//

void load_particle_data(){
    ifstream f("picdata.bin",std::ios::binary);
    if (f.fail()) {cout<<">> eduPIC: ERROR: No particle data file found, try running initial cycle using argument '0'"<<endl; exit(0); }
    f.read(reinterpret_cast<char*>(&Time),sizeof(double));
    f.read(reinterpret_cast<char*>(&cycles_done),sizeof(int));
    f.read(reinterpret_cast<char*>(&N_e),sizeof(int));
    x_e.resize(N_e);
    vx_e.resize(N_e);
    vy_e.resize(N_e);
    vz_e.resize(N_e);
    f.read(reinterpret_cast<char*>(&x_e[0]),N_e*sizeof(double));
    f.read(reinterpret_cast<char*>(&vx_e[0]),N_e*sizeof(double));
    f.read(reinterpret_cast<char*>(&vy_e[0]),N_e*sizeof(double));
    f.read(reinterpret_cast<char*>(&vz_e[0]),N_e*sizeof(double));
    f.read(reinterpret_cast<char*>(&N_i),sizeof(int));
    x_i.resize(N_i);
    vx_i.resize(N_i);
    vy_i.resize(N_i);
    vz_i.resize(N_i);
    f.read(reinterpret_cast<char*>(&x_i[0]),N_i*sizeof(double));
    f.read(reinterpret_cast<char*>(&vx_i[0]),N_i*sizeof(double));
    f.read(reinterpret_cast<char*>(&vy_i[0]),N_i*sizeof(double));
    f.read(reinterpret_cast<char*>(&vz_i[0]),N_i*sizeof(double));
    f.close();

    cout<<">> eduPIC: data loaded : "<<N_e<<" electrons "<<N_i<<" ions, "
    <<cycles_done<<" cycles completed before, time is "<<scientific<<Time<<" [s]"<<endl;
}

//---------------------------------------------------------------------//
// save density data                                                   //
//---------------------------------------------------------------------//

void save_density(void){
    ofstream f("density.dat");
    f<<setprecision(12)<<fixed<<scientific;

    auto c=1.0/static_cast<double>(no_of_cycles)/static_cast<double>(N_T);
    for(size_t i{0}; i<N_G;++i){
        f<<i*DX<<" "<<cumul_e_density[i]*c<<" "<<cumul_i_density[i]*c<<endl;
    }
    f.close();
}

//---------------------------------------------------------------------//
// save EEPF data                                                      //
//---------------------------------------------------------------------//

void save_eepf(void) {
    ofstream f("eepf.dat");
    auto h=accumulate(eepf.begin(),eepf.end(),0.0);
    h *= DE_EEPF;
    f<<scientific;
    double energy {};
    for(size_t i{0}; i<N_EEPF;++i){
        energy=(i + 0.5) * DE_EEPF;
        f<<energy<<" "<<eepf[i] / h / sqrt(energy)<<endl;
    }
    f.close();
}

//---------------------------------------------------------------------//
// save IFED data                                                      //
//---------------------------------------------------------------------//

void save_ifed(void) {
    double p, g, energy;
    ofstream f("ifed.dat");
    f<<scientific;
    double h_pow = accumulate(ifed_pow.begin(),ifed_pow.end(),0.0);
    double h_gnd = accumulate(ifed_gnd.begin(),ifed_gnd.end(),0.0);
    h_pow *= DE_IFED;
    h_gnd *= DE_IFED;
    mean_i_energy_pow = 0.0;
    mean_i_energy_gnd = 0.0;
    for(size_t i{0}; i<N_IFED;++i){
        energy = (i + 0.5) * DE_IFED;
        p = static_cast<double>(ifed_pow[i]) / h_pow;
        g = static_cast<double>(ifed_gnd[i]) / h_gnd;
        f<<energy<<" "<<p<<" "<<g<<endl;
        mean_i_energy_pow += energy * p;
        mean_i_energy_gnd += energy * g;
    }
    f.close();
}

//--------------------------------------------------------------------//
// save XT data                                                       //
//--------------------------------------------------------------------//

void save_xt_1(xt_distr &distr, string fname) {
    ofstream f(fname);
    ostream_iterator<double> tof(f," ");
    auto it=distr.begin();

    f<<setprecision(8)<<fixed<<scientific;
    for(size_t i{0};i<N_G;++i){
        copy_n(it,N_XT,tof);
        advance(it,N_XT);
        f<<endl;
    }
    f.close();
}



void norm_all_xt(void){    
    // normalize all XT data
    
    double f1 = static_cast<double>(N_XT) / static_cast<double>(no_of_cycles * N_T);
    double f2 = WEIGHT / (ELECTRODE_AREA * DX) / (no_of_cycles * (PERIOD / static_cast<double>(N_XT)));
    
    transform(pot_xt.begin(),pot_xt.end(),pot_xt.begin(),[=](auto y){return f1*y;});
    transform(efield_xt.begin(),efield_xt.end(),efield_xt.begin(),[=](auto y){return f1*y;});
    transform(ne_xt.begin(),ne_xt.end(),ne_xt.begin(),[=](auto y){return f1*y;});
    transform(ni_xt.begin(),ni_xt.end(),ni_xt.begin(),[=](auto y){return f1*y;});

    transform(ue_xt.begin(),ue_xt.end(),counter_e_xt.begin(),ue_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
    transform(ue_xt.begin(),ue_xt.end(),ne_xt.begin(),je_xt.begin(),[=](auto x, auto y){return -x*y*E_CHARGE;});
    transform(meanee_xt.begin(),meanee_xt.end(),counter_e_xt.begin(),meanee_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
    transform(ioniz_rate_xt.begin(),ioniz_rate_xt.end(),counter_e_xt.begin(),ioniz_rate_xt.begin(),[=](auto x, auto y){if(y>0){return x*f2;}else{return 0.0;}});

    transform(ui_xt.begin(),ui_xt.end(),counter_i_xt.begin(),ui_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
    transform(ui_xt.begin(),ui_xt.end(),ni_xt.begin(),ji_xt.begin(),[=](auto x, auto y){return x*y*E_CHARGE;});
    transform(meanei_xt.begin(),meanei_xt.end(),counter_i_xt.begin(),meanei_xt.begin(),[](auto x, auto y){if(y>0){return x/y;}else{return 0.0;}});
 
    transform(je_xt.begin(),je_xt.end(),efield_xt.begin(),powere_xt.begin(),[=](auto x, auto y){return x*y;});
    transform(ji_xt.begin(),ji_xt.end(),efield_xt.begin(),poweri_xt.begin(),[=](auto x, auto y){return x*y;});
}

    
void save_all_xt(void){
    
    save_xt_1(pot_xt, "pot_xt.dat");
    save_xt_1(efield_xt, "efield_xt.dat");
    save_xt_1(ne_xt, "ne_xt.dat");
    save_xt_1(ni_xt, "ni_xt.dat");
    save_xt_1(je_xt, "je_xt.dat");
    save_xt_1(ji_xt, "ji_xt.dat");
    save_xt_1(powere_xt, "powere_xt.dat");
    save_xt_1(poweri_xt, "poweri_xt.dat");
    save_xt_1(meanee_xt, "meanee_xt.dat");
    save_xt_1(meanei_xt, "meanei_xt.dat");
    save_xt_1(ioniz_rate_xt, "ioniz_xt.dat");
}

//---------------------------------------------------------------------//
// simulation report including stability and accuracy conditions       //
//---------------------------------------------------------------------//

void check_and_save_info(void){
    ofstream f("info.txt");
    string line (80,'-');
    f<<setprecision(4)<<fixed<<scientific;

    double density = cumul_e_density[N_G / 2]
                        / static_cast<double>(no_of_cycles) / static_cast<double>(N_T);            // e density @ center
    double plas_freq = E_CHARGE * sqrt(density / EPSILON0 / E_MASS);                               // e plasma frequency @ center
    double meane = mean_energy_accu_center / static_cast<double>(mean_energy_counter_center);      // e mean energy @ center
    double kT = 2.0 * meane * EV_TO_J / 3.0;                                                       // k T_e @ center (approximate)
    double debye_length = sqrt(EPSILON0 * kT / density) / E_CHARGE;                                // e Debye length @ center
    double sim_time =  static_cast<double>(no_of_cycles) / FREQUENCY;                              // simulated time
    double ecoll_freq =  static_cast<double>(N_e_coll) / sim_time /  static_cast<double>(N_e);     // e collision frequency
    double icoll_freq =  static_cast<double>(N_i_coll) / sim_time /  static_cast<double>(N_i);     // ion collision frequency

    f<<"########################## eduPIC simulation report ############################"<<endl;
    f<<"Simulation parameters:"<<endl;
    f<<"Gap distance                          = "<<L<<" [m]"<<endl;
    f<<"# of grid divisions                   = "<<N_G<<endl;
    f<<"Frequency                             = "<<FREQUENCY<<" [Hz]"<<endl;
    f<<"# of time steps / period              = "<<N_T<<endl;
    f<<"# of electron / ion time steps        = "<<N_SUB<<endl;
    f<<"Voltage amplitude                     = "<<VOLTAGE<<" [V]"<<endl;
    f<<"Pressure (Ar)                         = "<<PRESSURE<<" [Pa]"<<endl;
    f<<"Temperature                           = "<<TEMPERATURE<<" [K]"<<endl;
    f<<"Superparticle weight                  = "<<WEIGHT<<endl;
    f<<"# of simulation cycles in this run    = "<<no_of_cycles<<endl;
    f<<line<<endl;
    f<<"Plasma characteristics:"<<endl;  
    f<<"Electron density @ center             = "<<density<<" [m^{-3}]"<<endl;
    f<<"Plasma frequency @ center             = "<<plas_freq<<" [rad/s]"<<endl;
    f<<"Debye length @ center                 = "<<debye_length<<" [m]"<<endl;
    f<<"Electron collision frequency          = "<<ecoll_freq<<" [1/s]"<<endl;
    f<<"Ion collision frequency               = "<<icoll_freq<<" [1/s]"<<endl;
    f<<line<<endl;
    f<<"Stability and accuracy conditions:"<<endl;  
    auto conditions_OK = true;
    auto c = plas_freq * DT_E;
    f<<"Plasma frequency @ center * DT_e      = "<<c<<" (OK if less than 0.20)"<<endl;
    if (c > 0.2) {conditions_OK = false;}
    c = DX / debye_length;
    f<<"DX / Debye length @ center            = "<<c<<" (OK if less than 1.00)"<<endl;
    if (c > 1.0) {conditions_OK = false;}
    c = max_electron_coll_freq() * DT_E;   
    f<<"Max. electron coll. frequency * DT_E  = "<<c<<" (OK if less than 0.05"<<endl;
    if (c > 0.05) {conditions_OK = false;}
    c = max_ion_coll_freq() * DT_I;
    f<<"Max. ion coll. frequency * DT_I       = "<<c<<" (OK if less than 0.05)"<<endl;
    if (c > 0.05) {conditions_OK = false;}
    if (conditions_OK == false){
        f<<line<<endl;
        f<<"** STABILITY AND ACCURACY CONDITION(S) VIOLATED - REFINE SIMULATION SETTINGS! **"<<endl;
        f<<line<<endl;
        f.close();
        f<<">> eduPIC: ERROR: STABILITY AND ACCURACY CONDITION(S) VIOLATED! "<<endl;
        f<<">> eduPIC: for details see 'info.txt' and refine simulation settings!"<<endl;
    }
    else{
        // calculate maximum energy for which the Courant condition holds:
        double v_max = DX / DT_E;
        double e_max = 0.5 * E_MASS * v_max * v_max / EV_TO_J;
        f<<"Max e- energy for CFL     condition   = "<<e_max<<endl;
        f<<"Check EEPF to ensure that CFL is fulfilled for the majority of the electrons!"<<endl;
        f<<line<<endl;

        // saving of the following data is done here as some of the further lines need data
        // that are computed / normalized in these functions

        cout<<">> eduPIC: saving diagnostics data"<<endl;
        save_density();
        save_eepf();
        save_ifed();
        norm_all_xt();
        save_all_xt();
        f<<"Particle characteristics at the electrodes:"<<endl;
        f<<"Ion flux at powered electrode         = "<<N_i_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        f<<"Ion flux at grounded electrode        = "<<N_i_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        f<<"Mean ion energy at powered electrode  = "<<mean_i_energy_pow<<" [eV]"<<endl;
        f<<"Mean ion energy at grounded electrode = "<<mean_i_energy_gnd<<" [eV]"<<endl;
        f<<"Electron flux at powered electrode    = "<<N_e_abs_pow * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        f<<"Electron flux at grounded electrode   = "<<N_e_abs_gnd * WEIGHT / ELECTRODE_AREA / (no_of_cycles * PERIOD)<<" [m^{-2} s^{-1}]"<<endl;
        
        // calculate spatially and temporally averaged power absorption by the electrons and ions
        

        auto power_e = accumulate(powere_xt.begin(), powere_xt.end(), 0.0) / static_cast<double>(N_XT * N_G);
        auto power_i = accumulate(poweri_xt.begin(), poweri_xt.end(), 0.0) / static_cast<double>(N_XT * N_G);
        f<<line<<endl;
        f<<"Absorbed power calculated as <j*E>:"<<endl;
        f<<"Electron power density (average)      = "<<power_e<<" [W m^{-3}]"<<endl;
        f<<"Ion power density (average)           = "<<power_i<<" [W m^{-3}]"<<endl;
        f<<"Total power density (average)         = "<<power_e+power_i<<" [W m^{-3}]"<<endl;
        f<<line<<endl;
        f.close();  

    }
}

//------------------------------------------------------------------------------------------//
// main                                                                                     //
// command line arguments:                                                                  //
// [1]: number of cycles (0 for init)                                                       //
// [2]: "m" turns on data collection and saving                                             //
//------------------------------------------------------------------------------------------//

int main (int argc, char *argv[]){
    cout<<">> eduPIC: starting..."<<endl;
    cout<<">> eduPIC: **************************************************************************"<<endl;
    cout<<">> eduPIC: Copyright (C) 2021 Z. Donko et al."<<endl;
    cout<<">> eduPIC: This program comes with ABSOLUTELY NO WARRANTY"<<endl;
    cout<<">> eduPIC: This is free software, you are welcome to use, modify and redistribute it"<<endl;
    cout<<">> eduPIC: according to the GNU General Public License, https://www.gnu.org/licenses/"<<endl;
    cout<<">> eduPIC: **************************************************************************"<<endl;

    if (argc == 1) {
        cout<<">> eduPIC: error = need starting_cycle argument"<<endl;
        return 1;
    } else {
        vector<string> argList(argv+1,argv+argc);
        arg1 = stoi(argList[0]);
        if (argc > 2) {
            if (argList[1]=="m"){
                measurement_mode = true;                            // measurements will be done
            } else {
                measurement_mode = false;
            }
        }
    }
    if (measurement_mode) {
        cout<<">> eduPIC: measurement mode: on"<<endl;
    } else {
        cout<<">> eduPIC: measurement mode: off"<<endl;
    }
    set_electron_cross_sections_ar();
    set_ion_cross_sections_ar();
    calc_total_cross_sections();
    //test_cross_sections(); return 1;

    if (arg1 == 0) {
        ifstream file("picdata.bin",std::ios::binary);
        if (file.good()) { file.close();
            cout<<">> eduPIC: Warning: Data from previous calculation are detected."<<endl;
            cout<<"           To start a new simulation from the beginning, please delete all output files before running ./eduPIC 0"<<endl;
            cout<<"           To continue the existing calculation, please specify the number of cycles to run, e.g. ./eduPIC 100"<<endl; 
            exit(0); 
        }
        no_of_cycles = 1;                                 
        cycle = 1;                                        // init cycle
        init(N_INIT);                                     // seed initial electrons & ions
        cout<<">> eduPIC: running initializing cycle"<<endl;
        Time = 0;
        do_one_cycle();
        cycles_done = 1;
    } else {
        no_of_cycles = arg1;                              // run number of cycles specified in command line
        load_particle_data();                             // read previous configuration from file
        cout<<">> eduPIC: running "<<no_of_cycles<<" cycle(s)"<<endl;
        for (cycle=cycles_done+1;cycle<=cycles_done+no_of_cycles;cycle++) {do_one_cycle();}
        cycles_done += no_of_cycles;
    }
    datafile.close();
    save_particle_data();
    if (measurement_mode) {
        check_and_save_info();
    }
    cout<<">> eduPIC: simulation of "<<no_of_cycles<<" cycle(s) is completed."<<endl;
}
