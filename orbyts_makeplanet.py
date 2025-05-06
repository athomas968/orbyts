# SPONCHpop project
# Planet Formation Module
#
# by Anna Thomas
# 2023-2025
"""
Created on Wed Aug 23 11:55:44 2023

@author: annathomas
"""

# packages
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# local modules
import orbyts_disk as ds

# =============================================================================
# planet formation module
# =============================================================================

class planet:                          

    """
    A class that represents a planet. It simulates the evolution of the planet by updating its attributes at each time step.
    
    Attributes:
        PDM (bool):                 switch for planetesimal driven migration
        migration (bool):           switch for type I and II migration
        peb_acc (bool):             switch for pebble accretion
        plan_acc (bool):            switch for planetesimal accretion
        gas_acc (bool):             switch for gas accretion (both envelope contraction and runaway gas accretion)

        disk (object):              disk model
        p (object):                 planet formation functions
        m_0 (float):                birth mass of the embryo, kg
        a_0 (float):                inital semi-major axis of the embryo, m
        t_0 (float):                birth time of the planet, yrs
        B (float):                  width of the feeding zone in hill spheres, dimensionless
        f_iso (float):              factor for pebble isolation, dimensionless

        m (float):                  mass of the planet at timestep, kg
        a (float):                  semi-major axis of the planet from host star at timestep, m

        t_final (float):            final time for the simulation, yrs
        dt_yrs (float):             time step for the simulation, yrs
        dt (float):                 time step for the simulation, s
        t_range (numpy.ndarray):    array of time values for the simulation

        key_vals (numpy.ndarray):   array to store key values at each time step
                                    [0] label, [1] m, [2] a, [3] dmdt, [4] dadt, [5] menv, [6] mcore, [7] r, [8] peb_to_pln_ratio, [9] dmdt_peb, [10] dmdt_pln, [11] dmdt_gas
        oth_vals (numpy.ndarray):   array to store other values at each time step.
                                    [0] e_orb (0), [1] i_orb (0), [2] J_rot (0 kg m^2 s^-1), [3] J_rot_plane (0 degrees)
        fzmod (numpy.ndarray):      array to store feeding zone boundary locations and modified surface density at each time step
                                    [0] time, [1] fz_in, [2] fz_out, [3] sigma_mod

        ind_peb (int):              index of the time step when the planet reaches pebble isolation mass.
        ind_con (int):              index of the time step when the planet stops Kelvin-Helmholtz contraction.
        ind_mig (int):              index of the time step when the planet changes migration regime from type I to type II

        cum_peb_mass (float):       cumulative mass of pebbles accreted by the planet
        cum_pln_mass (float):       cumulative mass of planetesimals accreted by the planet
    """

    def __init__(self, disk, a_0, t_0, m_0 = 1e-5 * ds.M_e, B = 4, p_ratio = 0.5, PDM = False, migration = True, peb_acc = True, plan_acc = True, gas_acc = True, grain_growth = False):
        self.PDM            = PDM
        self.migration      = migration
        self.peb_acc        = peb_acc
        self.plan_acc       = plan_acc
        self.gas_acc        = gas_acc
        self.grain_growth   = grain_growth

        self.disk           = disk                                                          # disk model       
        self.m_0            = m_0                                                           # kg, inital mass of embryo
        self.t_0            = t_0                                                           # yrs, birth time of planet
        self.a_0            = a_0                                                           # m, initial distance of embryo to central star
        self.B              = B                                                             # dimensionless, width of feeding zone in hill spheres
        self.p_ratio        = p_ratio                                                       # dimensionless, fraction of planetsimals adding to the mass of the envelope/core after peb_iso is reached
        
        self.m              = m_0                                                           # kg, initialising mass of planet
        self.a              = a_0                                                           # m, initialising distance of planet to central star

        self.st             = self.disk.get_st_ini(time = self.t_0, radius = self.a_0, status = 1) # dimensionless, initialising stokes number at birth location of planet
        
        self.t_final        = self.t_0 + 3e6                                                # yrs, final time of simulation
        self.dt_yrs         = 500                                                           # yrs, time step of simulation        
        self.dt             = self.dt_yrs * ds.yr                                           # s, time step of simulation
        self.t_range        = np.arange(self.t_0, self.t_final, self.dt_yrs)*ds.yr
        
        self.key_vals       = np.zeros([(len(self.t_range)) , 9])                          # array to store key values at each time step
        self.oth_vals       = np.zeros([(len(self.t_range)) , 4])                           # array to store other values at each time step
        
        self.key_vals[0,1]  = self.m_0                                                      # initialising mass of planet
        self.key_vals[0,2]  = self.a_0                                                      # initialising semi major axis of planet
        
        self.ind_peb        = None                                                          # index of the time step when the planet reaches pebble isolation mass
        self.ind_con        = None                                                          # index of the time step when the planet stops envelope contraction
        self.ind_mig        = None                                                          # index of the time step when the planet changes migration regime from type I to type II

        self.cum_peb_mass   = 0
        self.cum_pln_mass   = 0

        self.rad_grid                       = np.array(self.disk.rstruct())                 # radial grid of disk
        self.grid_sizes                     = np.diff(self.rad_grid)                        # grid widths
        self.r_grid_centres                 = 0.5*(self.rad_grid[1:] + self.rad_grid[:-1])  # locations of grid centres
        self.planetesimal_surface_density   = [self.disk.get_disk_plan(radius = r, status = self.disk.validate( radius = r , time = float(0) )) for r in self.r_grid_centres]   # planetesimal surface density at each grid centre
        self.planetesimal_grid              = np.vstack((self.rad_grid[:-1], self.planetesimal_surface_density)).T                                                              # planetesimal grid
        self.pebble_surface_density         = np.array([self.disk.get_pebsig(radius = r, time = float(0), status=1) for r in self.r_grid_centres])    

    def eq11(self, eta, rcap, vkep, temperature, radius):  
        eta =  (self.disk.get_cs(temperature)/vkep)**2
        return vkep * max(eta , ( (3 * rcap) / (2*radius) ) )
    
    def single_planet_case(self):
        """
        Simulates the evolution of a single planet in a disk, updating its attributes at each time step.
        """
        dmdt_pln                = 0
        stokes_array = np.array([self.disk.get_st_ini(time=self.t_range[0], radius=r, status=1)for r in self.r_grid_centres])
        for i, t in enumerate(self.t_range):
            percent_complete = (i / len(self.t_range)) * 100
            if i % (len(self.t_range) // 100) == 0: 
              print(f"Progress: {percent_complete:.0f}%")
            self.key_vals[i,1]   = self.m                                                                            # kg, mass of planet
            self.key_vals[i,2]   = self.a                                                                            # m, semi major axis of planet   

            validation           = self.disk.validate( radius = self.a , time = self.t_range[i] )
            if validation != 1:
                print('Radius out of bounds')
                break

            bindex               = np.digitize(self.a, self.r_grid_centres)   
            
            fkep                 = self.disk.get_Omega( self.a )                                                     # /s, keplerian frequency
            vkep                 = fkep * np.pi * 2 * self.a                                                         # m/s, keplarian velocity
            
            r                    = self.disk.get_mr(self.m)                                                          # m, radius of planet
            self.key_vals[i,7]   = r

            temp                 = self.disk.get_T(self.a, self.t_range[i] , validation )                            # K, tempurature of disk midplane at semi-major axis of planet
            temp_array           = np.array([self.disk.get_T(r, self.t_range[i], validation) for r in self.r_grid_centres])            # K, temperature of disk midplane at each grid centre
            
            if self.grain_growth == True:
                stokes           = self.disk.stokes_calculator(self.a, self.t_range[i], self.st, validation)                  # dimensionless, stokes number at location of planet
                self.st          = stokes
                stokes_array     = np.array([self.disk.stokes_calculator(r, self.t_range[i], stokes_array[j], validation) for j, r in enumerate(self.r_grid_centres)])
            elif self.grain_growth == False:
                self.st          = np.minimum(self.disk.get_St_turb(temp), self.disk.get_St_drft(self.a, temp))
                stokes_array     = np.array([np.minimum(self.disk.get_St_turb(temp), self.disk.get_St_drft(r, temp)) for r, temp in zip(self.r_grid_centres, temp_array)]) # dimensionless, stokes number at each grid centre

            hillrad              = self.disk.get_hr(self.m, self.a, validation)                                      # m, hill radius of planet
            eta                  = (self.disk.get_cs(temp)/vkep)**2                                                  # dimensionless, eta = (cs/vkep)^2

            peb_iso              = self.disk.get_iso_peb(self.a, temp, status = validation )                         # kg, eq 12, john chambers 2018
            
            def eq8(rcap): 
                vrel =  self.eq11(eta, rcap, vkep, temp, self.a)
                return (rcap/hillrad)**3 + (( (2 * self.a * vrel ) / (3 * hillrad * vkep)) * (rcap/hillrad)**2 ) - (8 * self.st)
            
            rcap_g               = hillrad                                                                           # initial guess for planet's pebble capture radius          
            rcap_peb             = fsolve(eq8, rcap_g)[0]                                                            # m, capture radius of planet for pebbles
            vrel                 = self.eq11(eta, rcap_peb, vkep, temp, self.a)                                      # m/s, relative velocity between pebbles and planet   
            fg                   = self.disk.get_fg(self.m, r, self.eq11(eta, hillrad, vkep, temp, self.a))          # gravitational focussing factor of planet
            rcap_plan            = r * fg**0.5                                                                       # m, capture radius of planet for planetesimals, chambers 2014

            radial_drift         = np.array([self.disk.get_vr(radius = rad, time = self.t_range[i], status = validation) for rad in self.r_grid_centres])
            twodee_in            = self.pebble_surface_density * radial_drift
            twodee_out           = np.roll(twodee_in, -1)
            twodee_out[-1]       = 0
            
            #   PLANETESIMAL ACCRETION TAKES PLACE UNTIL DEPELETION OF PLANETESIMALS, REGARDLESS OF PEBBLE ISOLATION MASS BEING REACHED
            
            if self.plan_acc == True:
                modded_grid, modified_sigma = self.disk.update_plan_grid(self.a, self.planetesimal_grid, self.B, dmdt_pln, self.m, dt = self.dt, status = validation)
                dmdt_pln                    = self.disk.get_plnacc(modified_sigma, self.a, rcap_plan, fg)
            elif self.plan_acc == False:
                dmdt_pln                    = 0

            #   BELOW PEBBLE ISOLATION MASS - PEBBLE ACCRETION UNTIL ISOLATION MASS IS REACHED
            if (self.m < peb_iso) and self.a > (0.02 * ds.auSI):
                dmdt_gas                = 0 
                self.ind_peb            = i
                m_env                   = (self.key_vals[self.ind_peb,1] * 0.1)                                       # kg, mass of gas envelope, assumed to be 10% of planet mass SOURCE
                self.key_vals[i,5]      = m_env
                m_core                  = (self.key_vals[self.ind_peb,1] * 0.9)                                       # kg, mass of planet core, assumed to be 90% of planet mass
                self.key_vals[i,6]      = m_core
                self.cum_pln_mass       += dmdt_pln * self.dt

                if self.peb_acc == True:
                    dmdt_peb                            = self.disk.get_pebacc(rcap_peb, self.a, vrel, temp, self.st, self.pebble_surface_density[bindex])
                    pebble_deduct                       = dmdt_peb/self.grid_sizes[bindex]
                elif self.peb_acc == False:
                    dmdt_peb                            = 0
                    pebble_deduct                       = 0 #dmdt_peb/self.grid_sizes[bindex]
                
                self.key_vals[i,3]           = dmdt_peb + dmdt_pln + dmdt_gas
                self.m                      += (dmdt_peb + dmdt_gas + dmdt_pln) * self.dt 

            #   ABOVE PEBBLE ISOLATION MASS - ENVELOPE CONTRACTION AND POSSIBLE RUNAWAY GAS ACCRETION
            if (self.m >= peb_iso) and self.a > (0.02 * ds.auSI):
                m_core                       = (self.key_vals[self.ind_peb,1] * 0.9) + ((1-self.p_ratio) * dmdt_pln * self.dt)
                self.key_vals[i,6]           = m_core
                dmdt_peb                     = 0

                if self.gas_acc == True: 
                    if m_core > m_env:                                                                             # if mass of planet core exceeds mass of gas envelope, envelope contraction
                        self.ind_con         = i
                        dmdt_gas             = self.disk.get_envcon(m_core, m_env, temperature = temp, status = validation)
                    elif m_core <= m_env:                                                                           # if mass of planet gas envelope exceeds mass of core, runaway gas accretion
                        dmdt_gas             = self.disk.get_gasacc(self.t_range[i], self.a, self.m, temp, validation)
                elif self.gas_acc == False:
                    dmdt_gas = 0

                m_env                       += (dmdt_gas * self.dt) + (dmdt_pln * self.dt * self.p_ratio)
                self.key_vals[i,5]           = m_env
                self.key_vals[i,3]           = dmdt_peb + dmdt_pln + dmdt_gas
                self.m                       = m_core + m_env

            #   MIGRATION

            if self.migration == True:
                dadt                  = self.disk.get_mig(self.t_range[i], self.a, self.m, temp, status = validation)
                if hillrad < self.disk.get_scaleheight(self.a, temp):
                    self.ind_mig      = i
                if self.PDM == True:
                    dadt_PDM          = (self.disk.get_PDM(radius = self.a, sigma = modified_sigma, mass = self.m, bh_pdm = 2.2, mass_horseshoe = 1 * ds.M_e, status = validation)) * -ds.auSI/ds.yr
                elif self.PDM == False:
                    dadt_PDM          = 0
            elif self.migration == False:
                dadt                  = 0
                dadt_PDM              = 0

            if self.a < (0.02 * ds.auSI):       # if planet is within 0.02 AU of central star, migration and accretion stops
                dadt                  = 0
                dadt_PDM              = 0
                dmdt_gas              = 0
                dmdt_peb              = 0
                dmdt_pln              = 0

            if self.cum_pln_mass > 0:
                peb_to_pln_ratio = self.cum_peb_mass / self.cum_pln_mass
            else:
                peb_to_pln_ratio = float('inf')

            delta_sigma                         = (-twodee_in + twodee_out) / self.grid_sizes
            self.pebble_surface_density        += delta_sigma * self.dt
            self.pebble_surface_density[bindex] = max(0, (self.pebble_surface_density[bindex] - pebble_deduct))
            self.cum_peb_mass                  += dmdt_peb * self.dt

            self.key_vals[i,8]                  = peb_to_pln_ratio
            self.key_vals[i,4]                  = dadt + dadt_PDM
            self.a                             += self.key_vals[i,4] * self.dt

    def get_vals(self):
        return self.key_vals, self.t_range

    def get_other(self):
        return self.ind_peb, self.ind_con, self.ind_mig
