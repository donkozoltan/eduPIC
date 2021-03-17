//-------------------------------------------------------------------//
//       eduPIC : educational 1d3v PIC/MCC simulation code           //
//           version 1.0, release date: March 16, 2021               //
//                      :) Share & enjoy :)                          //
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

// switching off some compliler warnings
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_must_use)]
#![allow(unused_assignments)]

// include required modules
use rand::prelude::*;
use rand_distr::Normal;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::env;
use std::time::Instant;

// constants

const PI: f64              = 3.141592653589793;      // mathematical constant Pi
const TWO_PI: f64          = 2.0 * PI;               // two times Pi
const E_CHARGE: f64        = 1.60217662e-19;         // electron charge [C]
const EV_TO_J: f64         = E_CHARGE;               // eV <-> Joule conversion factor
const E_MASS: f64          = 9.10938356e-31;         // mass of electron [kg]
const AR_MASS: f64         = 6.63352090e-26;         // mass of argon atom [kg]
const MU_ARAR: f64         = AR_MASS / 2.0;          // reduced mass of two argon atoms [kg]
const K_BOLTZMANN: f64     = 1.38064852e-23;         // Boltzmann's constant [J/K]
const EPSILON0: f64        = 8.85418781e-12;         // permittivity of free space [F/m]

// simulation parameters

const N_G: usize           = 400;                    // number of grid points
const N_T: u32             = 4000;                   // time steps within an RF period
const FREQUENCY: f64       = 13.56e6;                // driving frequency [Hz]
const VOLTAGE: f64         = 250.0;                  // voltage amplitude [V]
const L: f64               = 0.025;                  // electrode L [m]
const PRESSURE: f64        = 10.0;                   // gas pressure [Pa]
const TEMPERATURE: f64     = 350.0;                  // background gas temperature [K]
const WEIGHT: f64          = 7.0e4;                  // weight of superparticles
const ELECTRODE_AREA: f64  = 1.0e-4;                 // (fictive) electrode area [m^2]
const N_INIT: usize        = 1000;                   // number of initial electrona and ions

// additional (derived) constants

const PERIOD: f64          = 1.0 / FREQUENCY;                          // RF period length [s]
const DT_E: f64            = PERIOD / (N_T as f64);                    // electron time step [s]
const N_SUB: u32           = 20;                                       // ions move only in these cycles (subcycling)
const DT_I: f64            = (N_SUB as f64) * DT_E;                    // ion time step [s]
const DX: f64              = L / ((N_G - 1) as f64);                   // spatial grid division [m]
const INV_DX: f64          = 1.0 / DX;                                 // inverse of spatial grid size [1/m]
const GAS_DENSITY: f64     = PRESSURE / (K_BOLTZMANN * TEMPERATURE);   // background gas gas density [m-3]
const OMEGA: f64           = TWO_PI * FREQUENCY;                       // angular frequency [rad/s]

// electron and ion cross sections

const N_CS: usize          = 5;                      // total number of processes / cross sections
const E_ELA: usize         = 0;                      // process identifier: electron/elastic
const E_EXC: usize         = 1;                      // process identifier: electron/excitation
const E_ION: usize         = 2;                      // process identifier: electron/ionization
const I_ISO: usize         = 3;                      // process identifier: ion/elastic/isotropic
const I_BACK: usize        = 4;                      // process identifier: ion/elastic/backscattering
const E_EXC_TH: f64        = 11.5;                   // electron impact excitation threshold [eV]
const E_ION_TH: f64        = 15.8;                   // electron impact ionization threshold [eV]
const CS_RANGES: usize     = 1_000_000;              // number of entries in cross section arrays
const DE_CS: f64           = 0.001;                  // energy division in cross section arrays [eV]

// measurement conditions

const MIN_X: f64           = 0.45 * L;               // lower limit of central region
const MAX_X: f64           = 0.55 * L;               // upper limit of central region
const N_EEPF: usize        = 2000;                   // number of energy bins in Electron Energy Probability Function (EEPF)
const DE_EEPF: f64         = 0.05;                   // resolution of EEPF [eV]
const N_FED: usize         = 200;                    // number of energy bins in Flux-Energy Distributions (EFED and IFED)
const DE_FED: f64          = 1.0;                    // resolution of FEDs (EFED and IFED) [eV]
const N_BIN: u32           = 20;                     // number of time steps binned for the XT distributions
const N_XT: usize          = (N_T / N_BIN) as usize; // number of spatial bins for the XT distributions

// structure type definitions

#[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
struct ParticleType {                                // coordinates of particles (one spatial, three velocity components)
    x: f64,                 
    vx: f64,
    vy: f64,
    vz: f64
}


//------------------------------------------------------------------------------------------//
// main                                                                                     //
// command line arguments:                                                                  //
// [1]: number of cycles (0 for init)                                                       //
// [2]: "m" turns on data collection and saving                                             //
//------------------------------------------------------------------------------------------//

fn main(){
    println!(">> eduPIC: starting...");
    println!(">> eduPIC: **************************************************************************");
    println!(">> eduPIC: Copyright (C) 2021 Z. Donko et al.");
    println!(">> eduPIC: This program comes with ABSOLUTELY NO WARRANTY");
    println!(">> eduPIC: This is free software, you are welcome to use, modify and redistribute it");
    println!(">> eduPIC: according to the GNU General Public License, https://www.gnu.org/licenses/");
    println!(">> eduPIC: **************************************************************************");
    // reading in command line arguments
    let args:Vec<String> = env::args().collect();
    if args.len() == 1 { println!(">> eduPIC: ERROR = need starting_cycle argument"); std::process::exit(1); }

    let cycle:usize = args[1].parse::<usize>().unwrap();
    let mut cycles_done:usize = 0;

    let mut measurement:bool = false;
    if (args.len() > 2) && (args[2].parse::<String>().unwrap() == "m") { measurement = true; }
    if measurement { println!(">> eduPIC: measurement mode: on"); } 
    else { println!(">> eduPIC: measurement mode: off"); }

    // initializing grid quantities (fixed size)
    let mut efield: Vec<f64>          = vec![0.0;N_G];       // electric field 
    let mut pot: Vec<f64>             = vec![0.0;N_G];       // electric potential
    let mut e_density: Vec<f64>       = vec![0.0;N_G];       // electron density
    let mut i_density: Vec<f64>       = vec![0.0;N_G];       // ion density
    let mut rho: Vec<f64>             = vec![0.0;N_G];       // charge density
    let mut cumul_e_density: Vec<f64> = vec![0.0;N_G];       // cumulative electron density
    let mut cumul_i_density: Vec<f64> = vec![0.0;N_G];       // cumulative ion density
    
    // Particle data
    let mut Electrons: Vec<ParticleType> = Vec::new();       // new empty vector for electrons
    let mut Ions: Vec<ParticleType>      = Vec::new();       // new empty vector for ions

    // cross section stuff
    let (sigma,sigma_tot_e,sigma_tot_i) = init_cross_sections();
    check_cross_sections(& sigma).map_err(|err| println!("{:?}", err)).ok();

    // measurement data buffers
    let mut N_e: usize         = 0;
    let mut N_i: usize         = 0;
    let mut N_e_abs_pow: u64   = 0;
    let mut N_e_abs_gnd: u64   = 0;
    let mut N_i_abs_pow: u64   = 0;
    let mut N_i_abs_gnd: u64   = 0;

    let mut eepf: Vec<f64>     = vec![0.0;N_EEPF];
    let mut efed_pow: Vec<u64> = vec![0;N_FED];
    let mut efed_gnd: Vec<u64> = vec![0;N_FED];
    let mut ifed_pow: Vec<u64> = vec![0;N_FED];
    let mut ifed_gnd: Vec<u64> = vec![0;N_FED];

    let mut pot_xt: Vec<Vec<f64>>           = vec![vec![0.0;N_G];N_XT];  // XT distribution of the potential
    let mut efield_xt: Vec<Vec<f64>>        = vec![vec![0.0;N_G];N_XT];  // XT distribution of the electric field
    let mut ne_xt: Vec<Vec<f64>>            = vec![vec![0.0;N_G];N_XT];  // XT distribution of the electron density
    let mut ni_xt: Vec<Vec<f64>>            = vec![vec![0.0;N_G];N_XT];  // XT distribution of the ion density
    let mut ue_xt: Vec<Vec<f64>>            = vec![vec![0.0;N_G];N_XT];  // XT distribution of the mean electron velocity
    let mut ui_xt: Vec<Vec<f64>>            = vec![vec![0.0;N_G];N_XT];  // XT distribution of the mean ion velocity
    let mut je_xt: Vec<Vec<f64>>            = vec![vec![0.0;N_G];N_XT];  // XT distribution of the electron current density
    let mut ji_xt: Vec<Vec<f64>>            = vec![vec![0.0;N_G];N_XT];  // XT distribution of the ion current density
    let mut powere_xt: Vec<Vec<f64>>        = vec![vec![0.0;N_G];N_XT];  // XT distribution of the electron powering (power absorption) rate
    let mut poweri_xt: Vec<Vec<f64>>        = vec![vec![0.0;N_G];N_XT];  // XT distribution of the ion powering (power absorption) rate
    let mut meanee_xt: Vec<Vec<f64>>        = vec![vec![0.0;N_G];N_XT];  // XT distribution of the mean electron energy
    let mut meanei_xt: Vec<Vec<f64>>        = vec![vec![0.0;N_G];N_XT];  // XT distribution of the mean ion energy
    let mut counter_e_xt: Vec<Vec<f64>>     = vec![vec![0.0;N_G];N_XT];  // XT counter for electron properties
    let mut counter_i_xt: Vec<Vec<f64>>     = vec![vec![0.0;N_G];N_XT];  // XT counter for ion properties
    let mut ioniz_rate_xt: Vec<Vec<f64>>    = vec![vec![0.0;N_G];N_XT];  // XT distribution of the ionisation rate

    let mut mean_energy_accu_center: f64 = 0.0;
    let mut mean_e_energy_pow: f64       = 0.0;
    let mut mean_e_energy_gnd: f64       = 0.0;
    let mut mean_i_energy_pow: f64       = 0.0;
    let mut mean_i_energy_gnd: f64       = 0.0;
    let mut N_center_mean_energy: u64    = 0;
    let mut N_e_coll: u64                = 0;
    let mut N_i_coll: u64                = 0;

    let mut conditions_OK: bool          = true;

    let mut conv_file = OpenOptions::new().append(true).create(true).open("conv.dat").unwrap();

    // start clock for performance measure
    let start = Instant::now();
    let mut rng = rand::thread_rng();       // ThreadRNG - autoseeded from memory entropy, HC-128 algorithm

    if cycle == 0 {
        if std::path::Path::new("picdata.bin").exists() {
            println!(">> eduPIC: Warning: Data from previous calculation are detected.");
            println!("           To start a new simulation from the beginning, please delete all output files before running ./eduPIC 0");
            println!("           To continue the existing calculation, please specify the number of cycles to run, e.g. ./eduPIC 100");
            std::process::exit(0); 
        }
        // initializing particles
        init_particles(N_INIT, L, &mut Electrons, &mut rng);
        init_particles(N_INIT, L, &mut Ions, &mut rng);

        println!(">> eduPIC: Running initializing cycle...");
        for t in 0..N_T{
            if t%1000==0 {
                println!(
                    "c = {:8}  t = {:8}  #e = {:8}  #i = {:8}",
                    1, t, Electrons.len(), Ions.len() );
            }

            get_density(&mut e_density, &mut cumul_e_density, &mut Electrons);
            if t % N_SUB == 0 { get_density(&mut i_density, &mut cumul_i_density, &mut Ions); }
            
            rho = e_density.iter().zip(i_density.iter()).map(|(x,y)| E_CHARGE*(y-x)).collect();
            solve_poisson(&mut pot, &mut efield, &rho, VOLTAGE * ((t as f64) / (N_T as f64) * TWO_PI).cos()); 

            move_particles(&efield, &mut Electrons, E_MASS, DT_E, -E_CHARGE);
            if t % N_SUB == 0 { move_particles(&efield, &mut Ions, AR_MASS, DT_I, E_CHARGE); }

            check_boundaries(&mut Electrons, &mut N_e_abs_pow, &mut N_e_abs_gnd, measurement, & mut efed_pow, &mut efed_gnd, E_MASS);
            if t % N_SUB == 0 { check_boundaries(&mut Ions, &mut N_i_abs_pow, &mut N_i_abs_gnd, measurement, & mut ifed_pow, &mut ifed_gnd, AR_MASS); }

            check_collisions_e(&mut Electrons, &mut Ions, &sigma_tot_e, &sigma, &mut N_e_coll, &mut rng);
            if t % N_SUB == 0 { check_collisions_i(&mut Ions, &sigma_tot_i, &sigma, &mut N_i_coll, &mut rng); }
        }

        cycles_done = 1;
        writeln!(conv_file, "{:10}   {:10}   {:10}", cycles_done, Electrons.len(), Ions.len()).map_err(|err| println!("{:?}", err)).ok();
        save_particle_data(String::from("picdata.bin"), cycles_done, &Electrons, &Ions).map_err(|err| println!("{:?}", err)).ok();

    } else {
        // load particles
        let(mut cycles_done, mut Electrons, mut Ions) = load_particle_data(String::from("picdata.bin"));

        println!(">> eduPIC: Running {} cycles...",cycle);
        for c in 1..=cycle{
            for t in 0..N_T{
                if t%1000==0{
                    println!(
                        "c = {:8}  t = {:8}  #e = {:8}  #i = {:8}",
                        cycles_done+c, t, Electrons.len(), Ions.len() );
                }

                get_density(&mut e_density, &mut cumul_e_density, &mut Electrons);
                if t % N_SUB == 0 { get_density(&mut i_density, &mut cumul_i_density, &mut Ions); }

                rho = e_density.iter().zip(i_density.iter()).map(|(x,y)| E_CHARGE*(y-x)).collect();
                solve_poisson(&mut pot, &mut efield, &rho, VOLTAGE * ((t as f64) / (N_T as f64) * TWO_PI).cos()); 

                move_particles(&efield, &mut Electrons, E_MASS, DT_E, -E_CHARGE);
                if t % N_SUB == 0 { move_particles(&efield, &mut Ions, AR_MASS, DT_I, E_CHARGE); }
    
                check_boundaries(&mut Electrons, &mut N_e_abs_pow, &mut N_e_abs_gnd, measurement, & mut efed_pow, &mut efed_gnd, E_MASS);
                if t % N_SUB == 0 { check_boundaries(&mut Ions, &mut N_i_abs_pow, &mut N_i_abs_gnd, measurement, & mut ifed_pow, &mut ifed_gnd, AR_MASS); }
    
                check_collisions_e(&mut Electrons, &mut Ions, &sigma_tot_e, &sigma, &mut N_e_coll, &mut rng);
                if t % N_SUB == 0 { check_collisions_i(&mut Ions, &sigma_tot_i, &sigma, &mut N_i_coll, &mut rng); }
 
                if measurement {
                    let t_index: usize = (t / N_BIN) as usize;
                    let mut p: usize;
                    let mut c1: f64;
                    let mut c2: f64;
                    let mut e_x: f64;
                    let mut mean_v: f64;
                    let mut v_sqr: f64;
                    let mut energy: f64;
                    let mut rate: f64;
                    let mut energy_index: usize;
                
                    // collect data from electrons: mean energy, mean velocity, ionization rate, EEPF
                    for part in Electrons.iter() {
                        p   = (part.x * INV_DX).trunc() as usize;
                        c2  = part.x * INV_DX - (p as f64);
                        c1  = 1.0-c2;
                        e_x = c1 * efield[p] + c2 * efield[p+1];
                        mean_v = part.vx - 0.5 * e_x * DT_E * E_CHARGE / E_MASS;
                        counter_e_xt[t_index][p]   += c1;
                        counter_e_xt[t_index][p+1] += c2;
                        ue_xt[t_index][p]   += c1 * mean_v;
                        ue_xt[t_index][p+1] += c2 * mean_v;
                        v_sqr  = mean_v * mean_v + part.vy * part.vy + part.vz * part.vz; 
                        energy = 0.5 * E_MASS * v_sqr / EV_TO_J;
                        meanee_xt[t_index][p]   += c1 * energy;
                        meanee_xt[t_index][p+1] += c2 * energy;
                        energy_index = core::cmp::min((energy / (DE_CS as f64) + 0.5).trunc() as usize, (CS_RANGES-1) as usize);
                        rate = sigma[E_ION][energy_index] * v_sqr.sqrt() * DT_E * GAS_DENSITY;
                        ioniz_rate_xt[t_index][p]   += c1 * rate;
                        ioniz_rate_xt[t_index][p+1] += c2 * rate;
                                            
                        if (MIN_X < part.x) && (part.x < MAX_X){
                            let e_index: usize = (energy / DE_EEPF).trunc() as usize;
                            if e_index < N_EEPF { eepf[e_index] += 1.0; }
                            mean_energy_accu_center += energy;
                            N_center_mean_energy    += 1;
                        }
                    }

                    // collect data from ions: mean energy, mean velocity
                    if t % N_SUB == 0 {
                        for part in Ions.iter() {
                            p   = (part.x * INV_DX).trunc() as usize;
                            c2  = part.x * INV_DX - (p as f64);
                            c1  = 1.0-c2;
                            e_x = c1 * efield[p] + c2 * efield[p+1];
                            mean_v = part.vx + 0.5 * e_x * DT_I * E_CHARGE / AR_MASS;
                            counter_i_xt[t_index][p]   += c1;
                            counter_i_xt[t_index][p+1] += c2;
                            ui_xt[t_index][p]   += c1 * mean_v;
                            ui_xt[t_index][p+1] += c2 * mean_v;
                            v_sqr  = mean_v * mean_v + part.vy * part.vy + part.vz * part.vz; 
                            energy = 0.5 * AR_MASS * v_sqr / EV_TO_J;
                            meanei_xt[t_index][p]   += c1 * energy;
                            meanei_xt[t_index][p+1] += c2 * energy;
                        }
                    }                    
                    // collect data from the grid
                    for i in 0..N_G {
                        pot_xt   [t_index][i] += pot[i];
                        efield_xt[t_index][i] += efield[i];
                        ne_xt    [t_index][i] += e_density[i];
                        ni_xt    [t_index][i] += i_density[i];
                    }
                }
            }
            writeln!(conv_file, "{:10}   {:10}   {:10}", cycles_done+c, Electrons.len(), Ions.len()).map_err(|err| println!("{:?}", err)).ok();
        }
        cycles_done += cycle;
        N_e = Electrons.len();
        N_i = Ions.len();
        save_particle_data(String::from("picdata.bin"), cycles_done, &Electrons, &Ions).map_err(|err| println!("{:?}", err)).ok();
    }

    if measurement {
        let norm: f64 = (N_XT as f64) / (cycle as f64) / (N_T as f64);
        calc_current_and_power(&mut je_xt, &mut powere_xt, &ue_xt, &ne_xt, &counter_e_xt, &efield_xt, -E_CHARGE, norm).map_err(|err| println!("{:?}", err)).ok();
        calc_current_and_power(&mut ji_xt, &mut poweri_xt, &ui_xt, &ni_xt, &counter_i_xt, &efield_xt, E_CHARGE, norm).map_err(|err| println!("{:?}", err)).ok();
        calc_fed(&efed_pow, &efed_gnd, &mut mean_e_energy_pow, &mut mean_e_energy_gnd).map_err(|err| println!("{:?}", err)).ok();
        calc_fed(&ifed_pow, &ifed_gnd, &mut mean_i_energy_pow, &mut mean_i_energy_gnd).map_err(|err| println!("{:?}", err)).ok();
        check_and_save_info(cumul_e_density[N_G/2], cycle, mean_energy_accu_center, N_center_mean_energy, 
            N_e, N_i, N_e_coll, N_i_coll, &sigma_tot_e, &sigma_tot_i, mean_e_energy_pow, mean_e_energy_gnd,
            mean_i_energy_pow, mean_i_energy_gnd, N_e_abs_pow, N_e_abs_gnd, N_i_abs_pow, N_i_abs_gnd, 
            cycle, &powere_xt, &poweri_xt, &mut conditions_OK).map_err(|err| println!("{:?}", err)).ok();
        if conditions_OK {    
            println!(">> eduPIC: Saving measurements to disk...");
            println!(">> saving densities.dat");
            save_densities(&cumul_e_density, &cumul_i_density, cycle).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving eepf.dat");
            save_eepf(&eepf).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving efed.dat");
            save_fed("efed.dat", &efed_pow, &efed_gnd).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving ifed.dat");
            save_fed("ifed.dat", &ifed_pow, &ifed_gnd).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving pot_xt.dat");
            save_xt_1("pot_xt.dat", &pot_xt, norm).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving efield_xt.dat");
            save_xt_1("efield_xt.dat", &efield_xt, norm).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving ne_xt.dat");
            save_xt_1("ne_xt.dat", &ne_xt, norm).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving ni_xt.dat");
            save_xt_1("ni_xt.dat", &ni_xt, norm).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving je_xt.dat");
            save_xt_1("je_xt.dat", &je_xt, 1.0).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving ji_xt.dat");
            save_xt_1("ji_xt.dat", &ji_xt, 1.0).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving powere_xt.dat");
            save_xt_1("powere_xt.dat", &powere_xt, 1.0).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving poweri_xt.dat");
            save_xt_1("poweri_xt.dat", &poweri_xt, 1.0).map_err(|err| println!("{:?}", err)).ok();
            let c: f64 = (WEIGHT / ELECTRODE_AREA / DX) / ((cycle as f64) * PERIOD / (N_XT as f64));
            println!(">> saving ioniz_xt.dat");
            save_xt_1("ioniz_xt.dat", &ioniz_rate_xt, c).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving meanee_xt.dat");
            save_xt_2("meanee_xt.dat", &meanee_xt, &counter_e_xt).map_err(|err| println!("{:?}", err)).ok();
            println!(">> saving meanei_xt.dat");
            save_xt_2("meanei_xt.dat", &meanei_xt, &counter_i_xt).map_err(|err| println!("{:?}", err)).ok();
        }
    }
    println!(">> eduPIC: Simulation of {} cycle(s) is completed lasting {:.3} sec.", cycle, 0.001*start.elapsed().as_millis() as f64);
}

//----------------------------------------------------------------------//
// move particles in E-field                                            //
//----------------------------------------------------------------------//

fn move_particles(efield:&Vec<f64>, Particle:&mut Vec<ParticleType>, mass:f64, dt:f64, charge:f64)
{
    let factor:f64 = dt / mass * charge;
    let mut p:usize;
    let mut e_x:f64;
    let mut c2:f64;
    for part in Particle.iter_mut() {
        p   = (part.x * INV_DX).trunc() as usize;
        c2  = part.x * INV_DX - (p as f64);
        e_x = (1.0-c2) * efield[p] + c2 * efield[p+1];
        part.vx += e_x * factor;
        part.x  += part.vx * dt;
    }
}


//----------------------------------------------------------------------//
// Ar+ / Ar collision                                                   //
//----------------------------------------------------------------------//

fn collision_ion(cs:&Vec<Vec<f64>>, Particle:&mut Vec<ParticleType>, 
    vxa:f64, vya:f64, vza:f64, particle_index:usize, eindex: usize, rng: &mut ThreadRng) 
{
    let t0 = cs[I_ISO][eindex];
    let t1 = t0 + cs[I_BACK][eindex];

    let phi: f64;
    let theta: f64;
    let chi: f64;
    let eta: f64;

    let mut gx = Particle[particle_index].vx - vxa; // relative velocity in cold gas approximation
    let mut gy = Particle[particle_index].vy - vya;
    let mut gz = Particle[particle_index].vz - vza;
    let g = ( gx.powf(2.0) + gy.powf(2.0) + gz.powf(2.0) ).sqrt();
    let wx = 0.5 * (Particle[particle_index].vx + vxa);
    let wy = 0.5 * (Particle[particle_index].vy + vya);
    let wz = 0.5 * (Particle[particle_index].vz + vza);

    // find Euler angles:
    if gx == 0.0 {
        theta = 0.5 * PI;
    } else {
        theta = ((gy * gy + gz * gz).sqrt()).atan2(gx);
    }
    if gy == 0.0 {
        if gz > 0.0 { phi = 0.5 * PI; } else { phi = -0.5 * PI; }
    } else {
        phi = gz.atan2(gy);
    }

    let rnd = rng.gen::<f64>();
    if rnd < t0 / t1 {
        chi = (1.0 - 2.0 * rng.gen::<f64>()).acos();
    } else {
        chi = PI;
    }
    eta = TWO_PI * rng.gen::<f64>();

    let sc = chi.sin();
    let cc = chi.cos();
    let se = eta.sin();
    let ce = eta.cos();
    let st = theta.sin();
    let ct = theta.cos();
    let sp = phi.sin();
    let cp = phi.cos();

    // compute new relative velocity:

    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);

    // post-collision velocity of the electron

    Particle[particle_index].vx = wx + 0.5 * gx;
    Particle[particle_index].vy = wy + 0.5 * gy;
    Particle[particle_index].vz = wz + 0.5 * gz;
}

fn check_collisions_i(Particle:&mut Vec<ParticleType>, total_cs_i:&Vec<f64>, cs:&Vec<Vec<f64>>,
    N_coll: &mut u64, rng: &mut ThreadRng) 
{
    let normal_range = Normal::new(0.0, (K_BOLTZMANN * TEMPERATURE / AR_MASS).sqrt()).unwrap();

    for k in 0..Particle.len() {
        let vxa = rng.sample(&normal_range);
        let vya = rng.sample(&normal_range);
        let vza = rng.sample(&normal_range);
        let gx = Particle[k].vx - vxa;
        let gy = Particle[k].vy - vxa;
        let gz = Particle[k].vz - vxa;
        let g2 = gx.powf(2.0) + gy.powf(2.0) + gz.powf(2.0);
        let g: f64 = g2.sqrt();
        let energy: f64 = 0.5 * MU_ARAR * g2 / EV_TO_J;
        let energy_index = core::cmp::min((energy / (DE_CS as f64) + 0.5).trunc() as usize, (CS_RANGES-1) as usize);

        let nu: f64 = total_cs_i[energy_index] * g;
        let p_coll: f64 = 1.0 - (-nu * DT_I).exp();
        if rng.gen::<f64>() < p_coll {
            collision_ion(cs, Particle, vxa, vya, vza, k, energy_index, rng);
            *N_coll += 1;
        }
    }
}


//----------------------------------------------------------------------//
// e / Ar collision                                                     //
//----------------------------------------------------------------------//

fn collision_electron(Electrons:&mut Vec<ParticleType>, Ions:&mut Vec<ParticleType>,
    cs:&Vec<Vec<f64>>, particle_index:usize, eindex: usize, rng: &mut ThreadRng) 
{
    let normal_range = Normal::new(0.0, (K_BOLTZMANN * TEMPERATURE / AR_MASS).sqrt()).unwrap();
    let f1 = E_MASS / (E_MASS + AR_MASS);
    let f2 = AR_MASS / (E_MASS + AR_MASS);
    let t0: f64 = cs[E_ELA][eindex];
    let t1: f64 = t0 + cs[E_EXC][eindex];
    let t2: f64 = t1 + cs[E_ION][eindex];
    
    let mut energy: f64 = 0.0;
    let mut e_new: f64  = 0.0;
    let mut e_orig: f64 = 0.0;

    let mut gx: f64 = Electrons[particle_index].vx;  // relative velocity in cold gas approximation
    let mut gy: f64 = Electrons[particle_index].vy;
    let mut gz: f64 = Electrons[particle_index].vz;
    let mut g: f64  = ( gx.powf(2.0) + gy.powf(2.0) + gz.powf(2.0) ).sqrt();
    let wx: f64 = f1 * gx;
    let wy: f64 = f1 * gy;
    let wz: f64 = f1 * gz;

    // find Euler angles:
    let phi: f64;
    let theta: f64;
    if gx == 0.0 { theta = 0.5 * PI; } 
    else { theta = ((gy * gy + gz * gz).sqrt()).atan2(gx); }
    if gy == 0.0 {
        if gz > 0.0 { phi = 0.5 * PI; } else { phi = -0.5 * PI; }
    } else { phi = gz.atan2(gy); }

    // choose the type of collision based on the cross sections
    // take into account energy loss in inelastic collisions
    // generate scattering and azimuth angles
    // in case of ionization handle the 'new' electron

    let chi: f64;
    let eta: f64;
    let mut sc: f64;
    let mut cc: f64;
    let mut se: f64;
    let mut ce: f64;
    let st: f64 = theta.sin();
    let ct: f64 = theta.cos();
    let sp: f64 = phi.sin();
    let cp: f64 = phi.cos();

    let rnd:f64 = rng.gen::<f64>();
    if rnd < t0 / t2 {                                    // elastic scattering
        chi = (1.0 - 2.0 * rng.gen::<f64>()).acos();      // isotropic scattering
        eta = TWO_PI * rng.gen::<f64>();                  // azimuthal angle
    } else if rnd < (t1 / t2) {                           // excitation
        energy = 0.5 * E_MASS * g.powf(2.0);              // electron energy
        energy = (energy - E_EXC_TH * EV_TO_J).abs();     // subtract energy loss for excitation
        g = (2.0 * energy / E_MASS).sqrt();               // relative velocity after energy loss
        chi = (1.0 - 2.0 * rng.gen::<f64>()).acos();      // isotropic scattering
        eta = TWO_PI * rng.gen::<f64>();                  // azimuthal angle
    } else {                                              // ionization
        energy = 0.5 * E_MASS * g.powf(2.0);              // electron energy
        energy = (energy - E_ION_TH * EV_TO_J).abs();     // subtract energy loss for ionization
        e_new  = 10.0 * (rng.gen::<f64>() * (energy/EV_TO_J/20.0).atan() ).tan() * EV_TO_J;
        e_orig = (energy - e_new).abs();                  // share energy according to: [Donko, Phys. Rev. E 57, 7126 (1998); Opal, J. Chem. Phys. 55, 4100 (1971)]
        g = (2.0 * e_orig / E_MASS).sqrt();               // relative velocity of incoming (original) electron
        let g_new: f64 = (2.0 * e_new / E_MASS).sqrt();   // relative velocity of emitted (new) electron
        chi = (e_orig / energy).sqrt().acos();            // scattering angle for incoming electron
        let chi_new: f64 = (e_new/ energy).sqrt().acos(); // scattering angle for emitted electron
        eta = TWO_PI * rng.gen::<f64>();                  // azimuthal angle for incoming electron
        let eta_new: f64 = eta + PI;                      // azimuthal angle for emitted electron
        sc = chi_new.sin();                               // scatter the emitted electron
        cc = chi_new.cos();
        se = eta_new.sin();
        ce = eta_new.cos();
        gx = g_new * (ct * cc - st * sc * ce);
        gy = g_new * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
        gz = g_new * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);
        Electrons.push(ParticleType{
            x : Electrons[particle_index].x,
            vx: (wx + f2 * gx),
            vy: (wy + f2 * gy),
            vz: (wz + f2 * gz)
        });
        Ions.push(ParticleType{
            x : Electrons[particle_index].x,
            vx: (rng.sample(&normal_range)),
            vy: (rng.sample(&normal_range)),
            vz: (rng.sample(&normal_range))
        });
    }

    // scatter the incoming electron
    
    sc = chi.sin();
    cc = chi.cos();
    se = eta.sin();
    ce = eta.cos();

    // compute new relative velocity:

    gx = g * (ct * cc - st * sc * ce);
    gy = g * (st * cp * cc + ct * cp * sc * ce - sp * sc * se);
    gz = g * (st * sp * cc + ct * sp * sc * ce + cp * sc * se);

    // post-collision velocity of the electron

    Electrons[particle_index].vx = wx + f2 * gx;
    Electrons[particle_index].vy = wy + f2 * gy;
    Electrons[particle_index].vz = wz + f2 * gz;
}

fn check_collisions_e(Electrons: &mut Vec<ParticleType>, Ions: &mut Vec<ParticleType>, 
    total_cs_e:&Vec<f64>, cs:&Vec<Vec<f64>>, N_coll: &mut u64, rng: &mut ThreadRng) 
{
    let N_e:usize = Electrons.len();
    for k in 0..N_e {
        let v2 = Electrons[k].vx.powf(2.0) + Electrons[k].vy.powf(2.0) + Electrons[k].vz.powf(2.0);
        let velocity: f64 = v2.sqrt();
        let energy: f64 = 0.5 * E_MASS * v2 / EV_TO_J;
        let energy_index = core::cmp::min((energy / (DE_CS as f64) + 0.5).trunc() as usize, (CS_RANGES-1) as usize);

        let nu: f64 = total_cs_e[energy_index] * velocity;
        let p_coll: f64 = 1.0 - (-nu * DT_E).exp();
        if rng.gen::<f64>() < p_coll {  
            collision_electron(Electrons, Ions, cs, k, energy_index, rng);
            *N_coll += 1;
        }
    }
}


//----------------------------------------------------------------------
// manage surface processes (absorption) 
//----------------------------------------------------------------------

fn check_boundaries(Particle: &mut Vec<ParticleType>, abs_pow: &mut u64, abs_gnd: &mut u64,
    measurement:bool, fed_pow : &mut Vec<u64>, fed_gnd: &mut Vec<u64>, mass: f64) 
{
    let mut ind:usize = 0;
    while ind < Particle.len(){
        let mut out:bool = false;
        if Particle[ind].x < 0.0 {          // the particle is out at the powered electrode
            *abs_pow += 1; 
            out = true; 
            if measurement {
                let v2: f64 = Particle[ind].vx.powf(2.0) + Particle[ind].vy.powf(2.0) + Particle[ind].vz.powf(2.0);
                let energy: f64 = 0.5 * mass * v2 / EV_TO_J;
                let energy_index: usize = (energy / (DE_FED as f64) + 0.5).trunc() as usize;
                if energy_index < N_FED {
                    fed_pow[energy_index] += 1;
                }
            }
        }   
        if Particle[ind].x > L   {          // the particle is out at the grounded electrode
            *abs_gnd += 1; 
            out = true; 
            if measurement {
                let v2: f64 = Particle[ind].vx.powf(2.0) + Particle[ind].vy.powf(2.0) + Particle[ind].vz.powf(2.0);
                let energy: f64 = 0.5 * mass * v2 / EV_TO_J;
                let energy_index: usize = (energy / (DE_FED as f64) + 0.5).trunc() as usize;
                if energy_index < N_FED {
                    fed_gnd[energy_index] += 1;
                }
            }
        }   

        if out{  
            Particle.swap_remove(ind);
        } else{ 
            ind += 1; 
        }
    }
}


//----------------------------------------------------------------------
// initialization routines
//----------------------------------------------------------------------

fn init_particles(np: usize, length: f64, Particle: &mut Vec<ParticleType>, rng: &mut ThreadRng) {
    let normal = Normal::new(0.0, (K_BOLTZMANN * TEMPERATURE / AR_MASS).sqrt()).unwrap();
    for _i in 0..np {
        let p0 = ParticleType {
            x: length * rng.gen::<f64>(),
            vx: rng.sample(normal),
            vy: rng.sample(normal),
            vz: rng.sample(normal),
        };
        Particle.push(p0);
    }
}


fn solve_poisson(pot: &mut Vec<f64>, efield: &mut Vec<f64>, rho: &Vec<f64>, pot0: f64) {
    const A: f64 = 1.0;
    const B: f64 = -2.0;
    const C: f64 = 1.0;
    const ALPHA: f64 = -DX * DX / EPSILON0;
    let mut h = vec![0.0; N_G];
    let mut w = vec![0.0; N_G];
    let mut f = vec![0.0; N_G];

    pot[0]     = pot0;
    pot[N_G-1] = 0.0;
    
    for i in 1..(N_G-1) { f[i] = ALPHA *rho[i]; }
    f[1] -= pot0;
    f[N_G - 2] -= pot[N_G-1];
    w[1] = C / B;
    h[1] = f[1] / B;
    for i in 2..(N_G-1) {
        w[i] = C / (B - A * w[i-1]);
        h[i] = (f[i] - A * h[i-1]) / (B - A * w[i-1]);
    }
    pot[N_G-2] = h[N_G-2];
    for i in (1..(N_G-2)).rev() {
        pot[i] = h[i] - w[i] * pot[i+1];
    }

    for i in 1..(N_G-1) { efield[i] = 0.5 * (pot[i-1] - pot[i+1]) * INV_DX; }
    efield[0]     = (pot[0]-pot[1]) * INV_DX - rho[0] * DX / (2.0 * EPSILON0);
    efield[N_G-1] = (pot[N_G-2]-pot[N_G-1]) * INV_DX + rho[N_G-1] * DX / (2.0 * EPSILON0);
}

//----------------------------------------------------------------------
// compute densitites from particle positions 
//----------------------------------------------------------------------

fn get_density(density:&mut Vec<f64>, cumul_density:&mut Vec<f64>, Particle:&Vec<ParticleType>)
{
    *density=vec![0.0;N_G];

    let c: f64 = WEIGHT / (ELECTRODE_AREA * DX);
    for p in Particle.iter(){
        let q:usize  = (p.x * INV_DX).trunc() as usize;
        let rem:f64  = p.x * INV_DX - (q as f64);
        density[q]   += (1.0-rem) * c;
        density[q+1] += rem * c;
    }
 
    density[0]     *= 2.0;
    density[N_G-1] *= 2.0;

    *cumul_density = cumul_density.iter().zip(density.iter()).map(|(x,y)| (y+x)).collect();
}


fn save_particle_data(filename: String, cycle_done:usize, Electrons:&Vec<ParticleType>, Ions:&Vec<ParticleType>)-> std::io::Result<()>
{
    let mut file = std::fs::File::create(filename).expect("unable to open file for writing");
    bincode::serialize_into(&mut file, &cycle_done).expect("unable to write to file");
    bincode::serialize_into(&mut file, &Electrons).expect("unable to write to file");
    bincode::serialize_into(&mut file, &Ions).expect("unable to write to file");
    Ok(())
}


fn load_particle_data(filename: String)->(usize,Vec<ParticleType>,Vec<ParticleType>)
{
    if !std::path::Path::new(&filename).exists() {println!(">> eduPIC: ERROR: No particle data file found, try running initial cycle using argument '0'"); std::process::exit(0); }
    let mut file = std::fs::File::open(filename).expect("unable to open file for reading");
    let c:usize                     = bincode::deserialize_from(&mut file).expect("unable to read from file");
    let Electrons:Vec<ParticleType> = bincode::deserialize_from(&mut file).expect("unable to read from file");
    let Ions:Vec<ParticleType>      = bincode::deserialize_from(&mut file).expect("unable to read from file");
    (c,Electrons,Ions)
}


fn init_cross_sections()->(Vec<Vec<f64>>,Vec<f64>,Vec<f64>) 
{
    let mut cs   = vec![vec![0.0;CS_RANGES];N_CS];
    let mut cs_i = vec![0.0;CS_RANGES];
    let mut cs_e = vec![0.0;CS_RANGES];
  
    let qmom = |e:f64| 1.0e-20*(
        (6.0/(1.0+e/0.1+(e/0.6).powf(2.0)).powf(3.3)-1.1*e.powf(1.4)/
        (1.0+(e/15.0).powf(1.2))/(1.0+(e/5.5).powf(2.5)+(e/60.0).powf(4.1)).sqrt()).abs()+0.05/(1.0+e/10.0).powf(2.0)+
        0.01*e.powf(3.0)/(1.0+(e/12.0).powf(6.0)));
  
    let qexc = |e:f64| { if e <= E_EXC_TH{0.0} else {(0.034 * (e - 11.5).powf(1.1) * (1.0 + (e / 15.0).powf(2.8))
        / (1.0 + (e / 23.0).powf(5.5)) + 0.023 * (e - 11.5) / (1.0 + e / 80.0).powf(1.9))*1.0e-20} };

    let qion = |e:f64| { if e <= E_ION_TH{0.0} else {(970.0 * (e - 15.8) / (70.0 + e).powf(2.0)
        + 0.06 * (e - 15.8).powf(2.0) * (-e / 9.0).exp())*1.0e-20 } };

    let qmoi = |e_lab:f64| 1.15e-18 * e_lab.powf(-0.1) * (1.0 + 0.015 / e_lab).powf(0.6); //2*e!
    let qiso = |e_lab:f64| 2.0e-19 * e_lab.powf(-0.5) / (1.0 + e_lab) + 3.0e-19 * e_lab / (1.0 + e_lab / 3.0).powf(2.0);
    let qchx = |e_lab:f64| 0.5*(qmoi(e_lab)-qiso(e_lab));


    let mut e_vec:Vec<f64>=(0..CS_RANGES).map(|x| (x as f64)*DE_CS).collect();
    e_vec[0] = DE_CS;

    cs[0]=e_vec.iter().zip(cs[0].iter()).map(|(&e,&c)| c+qmom(e)).collect();
    cs[1]=e_vec.iter().zip(cs[1].iter()).map(|(&e,&c)| c+qexc(e)).collect();
    cs[2]=e_vec.iter().zip(cs[2].iter()).map(|(&e,&c)| c+qion(e)).collect();
    cs[3]=e_vec.iter().zip(cs[3].iter()).map(|(&e,&c)| c+qiso(2.0*(e))).collect();
    cs[4]=e_vec.iter().zip(cs[4].iter()).map(|(&e,&c)| c+qchx(2.0*(e))).collect();

    for cs_s in (&cs[..3]).iter(){
        for (ind,val) in cs_s.iter().enumerate(){
            cs_e[ind]+=val*GAS_DENSITY;
        }
    }
    for cs_s in (&cs[3..]).iter(){
        for (ind,val) in cs_s.iter().enumerate(){
            cs_i[ind]+=val*GAS_DENSITY;
        }
    }  
    (cs,cs_e,cs_i)
}

// calculate mean impact energies
fn calc_fed(fed_pow:&Vec<u64>, fed_gnd:&Vec<u64>, mean_energy_pow:&mut f64, mean_energy_gnd:&mut f64)-> std::io::Result<()>
{
    let h_pow: f64 = (fed_pow.iter().sum::<u64>() as f64) * DE_FED;
    let h_gnd: f64 = (fed_gnd.iter().sum::<u64>() as f64) * DE_FED;
    *mean_energy_pow = 0.0;
    *mean_energy_gnd = 0.0;
    let mut energy:f64;
    for i in 0..N_FED {
        energy = (0.5 + i as f64) * DE_FED;
        *mean_energy_pow += energy * (fed_pow[i] as f64) / h_pow;
        *mean_energy_gnd += energy * (fed_gnd[i] as f64) / h_gnd;
    }
    Ok(())
}

fn calc_current_and_power(j_xt:&mut Vec<Vec<f64>>, pow_xt:&mut Vec<Vec<f64>>, u_xt:&Vec<Vec<f64>>, n_xt:&Vec<Vec<f64>>, c_xt:&Vec<Vec<f64>>, efield_xt:&Vec<Vec<f64>>, charge:f64, norm:f64)-> std::io::Result<()>
{
    let mut factor:f64;
    let mut u:f64;
    for i in 0..N_XT {
        for j in 0..N_G {
            factor = c_xt[i][j];
            if factor > 0.0 {factor = 1.0/factor;} else {factor = 0.0;}
            u = u_xt[i][j] * factor;
            j_xt[i][j]   = charge * u * n_xt[i][j] * norm;
            pow_xt[i][j] = j_xt[i][j] * efield_xt[i][j] * norm;
        }
    }
    Ok(())
}
// formatted output of cumulative densities
fn save_densities(eden:&Vec<f64>, iden:&Vec<f64>, cycle:usize)-> std::io::Result<()>
{
    let mut file = File::create("density.dat")?;
    for i in 0..N_G {
        writeln!(file, "{:1.6e} \t{:1.6e} \t{:1.6e}", (i as f64)*DX, eden[i] / (N_T as f64 * cycle as f64), iden[i] / ((N_T as f64 * cycle as f64 / N_SUB as f64) as f64) ); 
    }
    Ok(())
}

// save EEPF data 
fn save_eepf(eepf:&Vec<f64>)-> std::io::Result<()>
{
    let mut file = File::create("eepf.dat")?;
    let h: f64 = eepf.iter().sum::<f64>() * DE_EEPF;
    for i in 0..N_EEPF {
        let energy: f64 = (0.5 + i as f64) * DE_EEPF;
        writeln!(file, "{:1.6e} \t{:1.6e}", energy, eepf[i] / h / energy.sqrt() ); 
    }
    Ok(())
}

// save FED data 
fn save_fed(filename:&str, fed_pow:&Vec<u64>, fed_gnd:&Vec<u64>)-> std::io::Result<()>
{
    let mut file = File::create(filename)?;
    let h_pow: f64 = (fed_pow.iter().sum::<u64>() as f64) * DE_FED;
    let h_gnd: f64 = (fed_gnd.iter().sum::<u64>() as f64) * DE_FED;
    for i in 0..N_FED {
        let energy: f64 = (0.5 + i as f64) * DE_FED;
        let p = (fed_pow[i] as f64) / h_pow;
        let g = (fed_gnd[i] as f64) / h_gnd;
        writeln!(file, "{:1.6e} \t{:10} \t{:10}", energy, p, g); 
    }
    Ok(())
}

// save XT data  
fn save_xt_1(filename:&str, xt:&Vec<Vec<f64>>, norm:f64)-> std::io::Result<()>
{
    let mut file = File::create(filename)?;
    for j in 0..N_G {
        for i in 0..N_XT {
            write!(file, "{:1.6e}  ", xt[i][j] * norm); 
        }
        writeln!(file, "");
    }
    Ok(())
}

fn save_xt_2(filename:&str, xt:&Vec<Vec<f64>>, norm:&Vec<Vec<f64>>)-> std::io::Result<()>
{
    let mut file = File::create(filename)?;
    let mut factor:f64;
    for j in 0..N_G {
        for i in 0..N_XT {
            factor = norm[i][j];
            if factor > 0.0 {factor = 1.0/factor;} else {factor = 0.0;}
            write!(file, "{:1.6e}  ", xt[i][j] * factor); 
        }
        writeln!(file, "");
    }
    Ok(())
}

// formatted output of cross-sections for testing
fn check_cross_sections(s:& Vec<Vec<f64>>) -> std::io::Result<()>     // formatted output of cros-sections
{
    const N_SAVE:u32 = 1000;
    let mut file = File::create("cs.dat")?;
    let factor:f64 = ( (CS_RANGES as f64) ).powf(1.0/(N_SAVE as f64));
    for j in 1..N_SAVE {
        let en:f64  = DE_CS * factor.powf(j as f64);
        let i:usize = (en / DE_CS).trunc() as usize; 
        writeln!(file, "{:1.6e} \t{:1.6e} \t{:1.6e} \t{:1.6e} \t{:1.6e} \t{:1.6e}",
            en, s[0][i], s[1][i], s[2][i], s[3][i], s[4][i] ); 
    }
    Ok(())
}

// simulation report including stability and accuracy conditions       //
fn check_and_save_info(ne_max:f64, cycle:usize, mean_ee:f64, N_ee:u64, N_e:usize, N_i:usize, 
    N_e_coll:u64, N_i_coll:u64, total_cs_e:&Vec<f64>, total_cs_i:&Vec<f64>, mean_e_energy_pow:f64, 
    mean_e_energy_gnd:f64, mean_i_energy_pow:f64, mean_i_energy_gnd:f64, N_e_abs_pow:u64, N_e_abs_gnd:u64,
    N_i_abs_pow:u64, N_i_abs_gnd:u64, no_of_cycles:usize, powere_xt:&Vec<Vec<f64>>,
    poweri_xt:&Vec<Vec<f64>>, conditions_OK:&mut bool) -> std::io::Result<()>
{
    let mut file = File::create("info.txt")?;
    let density:f64      = ne_max / (cycle as f64) / (N_T as f64);                 // e density @ center
    let plas_freq:f64    = E_CHARGE * (density / EPSILON0 / E_MASS).sqrt();        // e plasma frequency @ center
    let meane:f64        = mean_ee / (N_ee as f64);                                // e mean energy @ center
    let kT:f64           = 2.0 * meane * EV_TO_J / 3.0;                            // k T_e @ center (approximate)
    let debye_length:f64 = (EPSILON0 * kT / density).sqrt() / E_CHARGE;            // e Debye length @ center
    let sim_time:f64     = (cycle as f64) / FREQUENCY;                             // simulated time
    let ecoll_freq:f64   = (N_e_coll as f64) / sim_time / (N_e as f64);            // e collision frequency
    let icoll_freq:f64   = (N_i_coll as f64) / sim_time / (N_i as f64);            // ion collision frequency

    // find upper limit of collision frequencies
    let mut max_ecoll_freq:f64 = 0.0;
    let mut max_icoll_freq:f64 = 0.0;
    let mut e:f64;
    let mut v:f64;
    let mut nu:f64;
    for i in 0..CS_RANGES{
        e  = (i as f64) * DE_CS;
        v  = (2.0 * e * EV_TO_J / E_MASS).sqrt();
        nu = v * total_cs_e[i];
        if nu > max_ecoll_freq { max_ecoll_freq = nu; }
        v  = (2.0 * e * EV_TO_J / MU_ARAR).sqrt();
        nu = v * total_cs_i[i];
        if nu > max_icoll_freq { max_icoll_freq = nu; }
    }

    writeln!(file, "########################## eduPIC simulation report ############################");
    writeln!(file, "Simulation parameters:");
    writeln!(file, "Gap distance                          = {:1.6e} [m]",  L );
    writeln!(file, "# of grid divisions                   = {:10}",        N_G );
    writeln!(file, "Frequency                             = {:1.6e} [Hz]", FREQUENCY );
    writeln!(file, "# of time steps / period              = {:10}",        N_T );
    writeln!(file, "# of electron / ion time steps        = {:10}",        N_SUB );
    writeln!(file, "Voltage amplitude                     = {:1.6e} [V]",  VOLTAGE );
    writeln!(file, "Pressure (Ar)                         = {:1.6e} [Pa]", PRESSURE );
    writeln!(file, "Temperature                           = {:1.6e} [K]",  TEMPERATURE );
    writeln!(file, "Superparticle weight                  = {:1.6e} [m]",  WEIGHT );
    writeln!(file, "# of simulation cycles in this run    = {:10}",        cycle );
    writeln!(file, "--------------------------------------------------------------------------------");
    writeln!(file, "Plasma characteristics:");
    writeln!(file, "Electron density @ center             = {:1.6e} [m^-3]",  density);
    writeln!(file, "Plasma frequency @ center             = {:1.6e} [rad/s]", plas_freq);
    writeln!(file, "Debye length @ center                 = {:1.6e} [m]",     debye_length);
    writeln!(file, "Electron collision frequency          = {:1.6e} [1/s]",   ecoll_freq);
    writeln!(file, "Ion collision frequency               = {:1.6e} [1/s]",   icoll_freq);
    writeln!(file, "--------------------------------------------------------------------------------");
    writeln!(file, "Stability and accuracy conditions:");
    *conditions_OK = true;
    let mut c:f64 = plas_freq * DT_E;
    writeln!(file, "Plasma frequency @ center * DT_E      = {:10.4} (OK if less than 0.20)", c);
    if c > 0.2 { *conditions_OK = false; }
    c = DX / debye_length;
    writeln!(file, "DX / Debye length @ center            = {:10.4} (OK if less than 1.00)", c);
    if c > 1.0 { *conditions_OK = false; }
    c = max_ecoll_freq * DT_E;
    writeln!(file, "Max. electron coll. frequency * DT_E  = {:10.4} (OK if less than 0.05)", c);
    if c > 0.05{ *conditions_OK = false; }
    c = max_icoll_freq * DT_I;
    writeln!(file, "Max. ion coll. frequency * DT_I       = {:10.4} (OK if less than 0.05)", c);
    if c > 0.05{ *conditions_OK = false; }
    if *conditions_OK == false {
        writeln!(file, "--------------------------------------------------------------------------------");
        writeln!(file, "** STABILITY AND ACCURACY CONDITION(S) VIOLATED - REFINE SIMULATION SETTINGS! **");
        writeln!(file, "--------------------------------------------------------------------------------");
        println!(">> eduPIC:  ERROR = STABILITY AND ACCURACY CONDITION(S) VIOLATED!");
        println!(">> eduPIC:  for details see 'info.txt' and refine simulation settings!");
    } else {

        // calculate maximum energy for which the Courant condition holds:
        let v_max:f64 = DX / DT_E;
        let e_max:f64 = 0.5 * E_MASS * v_max * v_max / EV_TO_J;
        writeln!(file, "Max e- energy for CFL condition       = {:10.4} [eV]", e_max);
        writeln!(file, "Check EEPF to ensure that CFL is fulfilled for the majority of the electrons!");
        writeln!(file, "--------------------------------------------------------------------------------");
        writeln!(file, "Particle characteristics at the electrodes:");
        writeln!(file, "Ion flux at powered electrode         = {:1.6e} [m^(-2) s^(-1)]", (N_i_abs_pow as f64) * WEIGHT / ELECTRODE_AREA / ((no_of_cycles as f64) * PERIOD));
        writeln!(file, "Ion flux at grounded electrode        = {:1.6e} [m^(-2) s^(-1)]", (N_i_abs_gnd as f64) * WEIGHT / ELECTRODE_AREA / ((no_of_cycles as f64) * PERIOD));
        writeln!(file, "Mean ion energy at powered electrode  = {:1.6e} [eV]", mean_i_energy_pow);
        writeln!(file, "Mean ion energy at grounded electrode = {:1.6e} [eV]", mean_i_energy_gnd);
        writeln!(file, "Electron flux at powered electrode    = {:1.6e} [m^(-2) s^(-1)]", (N_e_abs_pow as f64) * WEIGHT / ELECTRODE_AREA / ((no_of_cycles as f64) * PERIOD));
        writeln!(file, "Electron flux at grounded electrode   = {:1.6e} [m^(-2) s^(-1)]", (N_e_abs_gnd as f64) * WEIGHT / ELECTRODE_AREA / ((no_of_cycles as f64) * PERIOD));
        writeln!(file, "Mean electron energy at powered ele.  = {:1.6e} [eV]", mean_e_energy_pow);
        writeln!(file, "Mean electron energy at grounded ele. = {:1.6e} [eV]", mean_e_energy_gnd);
        writeln!(file, "--------------------------------------------------------------------------------");

        // calculate spatially and temporally averaged power absorption by the electrons and ions
        let mut power_e:f64 = 0.0;
        let mut power_i:f64 = 0.0;
        for i in 0..N_XT{
            for j in 0..N_G{
                power_e += powere_xt[i][j];
                power_i += poweri_xt[i][j];
            }
        }
        power_e /= (N_XT * N_G) as f64;
        power_i /= (N_XT * N_G) as f64;
        writeln!(file, "Absorbed power calculated as <j*E>:");
        writeln!(file, "Electron power density (average)      = {:1.6e} [W m^(-3)]", power_e);
        writeln!(file, "Ion power density (average)           = {:1.6e} [W m^(-3)]", power_i);
        writeln!(file, "Total power density(average)          = {:1.6e} [W m^(-3)]", power_e + power_i);
        writeln!(file, "--------------------------------------------------------------------------------\n");
        
    }
    Ok(())
}
