# SPONCHpop project
# Physical disk model
#
# by Mihkel Kama and Oliver Shorttle
# 2020

import sys
import numpy as np
import orbyts.physcon
from scipy import integrate

debug        = False

print("")
print("---------------------------")
print("---------------------------")
print("-- Disk structure module --")
print("---------------------------")
print("---------------------------")
print("   All units are SI unless stated otherwise.")
print("")

### Physical constants ###
M_sun                   = 1.989e+30            # kg
R_sun                   = 6.957e+8             # m
gamma                   = 1.4                  # adiabatic index
auSI                    = 1.496e+11            # m
yr                      = 365.2425*24*3600     # s
M_e                     = 5.972e24             # kg

class disk_master():

        def __init__(self, M_0=0.05*M_sun, alpha=1e-4, s_0=33*auSI, T_e=1380.0, M_star=1.0*M_sun, T_star=4200.0, R_star=1.0*R_sun, kappa_0=0.3, mu=2.4 , dtgr = 0.015, plnr = 0.25, pebr = 0.75, dustr = 0.005, vfrag = 2.5, rhos=1250):

                self.M_0            = M_0                  # starting mass of disk in kg
                self.alpha          = alpha                # shakura-sunyeav alpha paramater (efficiency of momenntum transport taking into account turbulence)
                self.s_0            = s_0                  # starting radius of disk in m
                if not 1e-2 < self.s_0/auSI < 1e+4:        print("!! s_0 value looks weird; should be given in [m]")
                if not 1e-6 < self.M_0/M_sun < 10:         print("!! M_0 value looks weird; should be given in [kg]")
                if not 0    < self.alpha < 1:              print("!! viscous alpha looks weird; was expecting 0<alpha<1")
                self.T_e            = T_e                  # sublimation temperature in K
                self.M_star         = M_star               # mass of star in kg
                self.T_star         = T_star               # temp of star in K
                self.R_star         = R_star               # radius of star in m
                self.kappa_0        = kappa_0              # opacity in m^2/kg
                self.mu             = mu                   # mean molecular weight
                self.dtgr           = dtgr                 # dust to gas ratio
                self.plnr           = plnr                 # ratio of planetesimals
                self.pebr           = pebr                 # ratio of pebbles        
                self.dustr          = dustr                # dust fraction
                self.vfrag          = vfrag                # pebble fragmentation speed, chambers 2018 popsyn, table 1, in m/s    
                self.rhos           = rhos                 # internal density of solids in kg/m^3 (based on Drazkowska+2021)

        # Eq. 11 -- Keplerian frequency [s^-1], orbital velocity
        def get_Omega(self, radius ):
                tempOmega = ( physcon.G*self.M_star/radius**3 )**(1./2.)
                if debug: print("debug:                Omega = %.3e s^-1 (P=%.3e years)" % ( tempOmega, 1.0/(yr*tempOmega) ) )
                return tempOmega

        # A logarithmic radial grid for the surface density evolution
        def rstruct(self):
                return [ 10**(0.01*elem) * auSI for elem in range(-160,241) ]

        def rstruct_chem(self): # lower resolution radial grid for chemical evolution of disk
                return [ 10**(0.01*elem) * auSI for elem in range(-160,241,50) ]      

        # The local sound speed for an ideal gas (default gamma=7./5. diatomic gas)
        def get_cs(self, temperature, gamma=7./5. ):
                tempcs = np.sqrt( gamma*physcon.k_B*temperature / physcon.m_p )
                return tempcs

        # Gas viscous radial velocity 
        def get_gasv(self, time, radius, status):  
            gasv = self.get_Mdot(time) / (2 * np.pi * radius  * self.get_Sigma(radius, time, status))
            return gasv
        
        def get_scaleheight(self, radius, temperature ):
            return self.get_cs( temperature ) / self.get_Omega( radius )
        
        # Method to validate disk structure (radius, time) values.
        # Unless we exit the code here, return
        #  0 initial value, this should never be returned
        #  1 pass, proceed as normal
        #  2 radius > r_out
        # The returned value can then be passed on to other functions so validate() doesn't need to be called everywhere all the time.
        # 2020.09.11: currently only implemented for get_T and get_Sigma, will expand on that as needed
        # 2021.14.08-OS: replaced type checks on time and radius with instance checks

        def validate(self, radius=None, time=None, status=0 ):
                if time != None:
                        if not isinstance(time, (np.floating, float)):
                                print('time=',time)
                                print('radius=',radius)
                                sys.exit("!! Time needs to be of type 'float', is " + str(type(time)))
                        elif time < 0:
                                sys.exit("!! Time t=%.3e yr < 0" % (time/yr) )
                        else:
                                status = 1
                if radius != None:
                        if not isinstance(radius, (np.floating, float)):
                                sys.exit("!! Radial location needs to be of type 'float', is " + str(type(radius)))
                        elif radius <= 0:
                                sys.exit("!! Radius r=%.3e au < 0" % (radius/auSI) )
                        elif radius > self.get_rout( time ):
                                status = 2
                        else:
                                status = 1
                if debug: print("debug:                status=%i for r=%s m, t=%s s" % ( status, str(radius), str(time) ) )
                return status
                     
class disk_viscirr(disk_master):
        print("-- Setting up a disk subclass: viscous and irradiated (Chambers 2009)")
        def __init__(self, **kwargs):
                super().__init__(**kwargs)
                #super().__init__(self, self.M_0, self.alpha, self.s_0, self.T_e, self.M_star, self.T_star, self.R_star, self.kappa_0, self.mu, self.dtgr, self.plnr, self.pebr, self.vfrag)
                # Eq. 30 -- potential equivalent temperature
                self.T_c                        = physcon.G*self.M_star*self.mu*physcon.m_p / ( physcon.k_B*self.R_star )
                if debug: print("debug:                T_c = %.2f K" % ( self.T_c ) )

                # Eq. 19 -- Sigma_0 gives Sigma_vis if r_e<r<r_t, which is the case in a pure viscous disk
                self.Sigma_0                    = 7.0*self.M_0 / ( 10.0*np.pi*self.s_0**2 )
                if debug: print("debug:                Sigma_0 = %.3e kg/m^2" % ( self.Sigma_0 ) )

                # Eq. 11 -- Keplerian frequency at s_0
                self.Omega_0                     = ( physcon.G*self.M_star/self.s_0**3 )**(1./2.)
                if debug: print("debug:                Omega_0 = %.3e s^-1 (P=%.3e years)" % ( self.Omega_0, 1.0/(yr*self.Omega_0) ) )

                # Eq. 20 -- T_vis at s_0; also T_0 == T_vis as defined in the pure viscous disk case
                self.T_vis                        = ( 27.0*self.kappa_0 / (64.0 * physcon.sigma) )**(1./3.) * ( self.alpha*gamma*physcon.k_B/(self.mu*physcon.m_p) )**(1./3.) * self.Sigma_0**(2./3.) * ( physcon.G*self.M_star/self.s_0**3 )**(1./6.)
                if debug: print("debug:                T_vis = %.2f K" % ( self.T_vis ) )

                # Eq. 32 -- midplane temperature in the irradiated region of the disk, t_rad
                self.T_rad                      = (4./7.)**(1./4.) * (self.T_star/self.T_c)**(1./7.) * (self.R_star/self.s_0)**(3./7.) * self.T_star
                if debug: print("debug:                T_rad = %.2f K" % ( self.T_rad ) )

                if self.T_vis >= self.T_rad:
                        self.Sigma_vis        = self.Sigma_0
                        self.T_0            = self.T_vis
                        self.Sigma_rad        = self.Sigma_vis * ( self.T_vis/self.T_rad )
                else:
                        self.Sigma_rad        = ( 13.0*self.M_0/(28.0*np.pi*self.s_0**2) ) * ( 1.0 - (33./98.)*(self.T_vis/self.T_rad)**(52./33.) )**(-1)
                        self.Sigma_0        = self.Sigma_rad
                        self.T_0            = self.T_rad
                        self.Sigma_vis        = self.Sigma_rad * ( self.T_rad/self.T_vis )**(4./5.)

                # Eq. 13 -- the initial viscosity at s_0
                self.nu_0                        = ( self.alpha*gamma*physcon.k_B / (self.mu*physcon.m_p) ) * self.T_0/self.Omega_0
                if debug: print("debug:                nu_0 = %.2f m^2/kg" % ( self.nu_0 ) )

                # Eq. 14 -- initial gas mass accretion rate (absolute value is constant through the disk, actual value transitions from positive to negative at some radius)
                self.M_0_dot                = 3.0 * np.pi * self.Sigma_0 * self.nu_0
                if debug: print("debug:                M_0_dot = %.3e Msun/yr" % ( self.M_0_dot*yr/M_sun ) )

                # Eq. 26 -- the viscous timescale
                self.tau_vis                = 3.0*self.M_0 / (16.0*self.M_0_dot) #self.s_0**2 / self.nu_0 
                if debug: print("debug:                tau_vis = %.3e yr" % (self.tau_vis/yr ) )

                # Eq. 52 -- time t_1 at which an irradiated zone first appears
                self.t_1                    = self.tau_vis * ( ( self.T_vis/self.T_rad )**(112./73.) - 1.0 )
                if debug: print("debug:                t_1 = %.3e yr" % (self.t_1/yr ) )

                # Eq. 51 -- disk mass at t_1
                self.M_1                    = self.M_0 * ( self.T_rad/self.T_vis )**(21./73.)

                # Eq. 50 -- outer radius at t_1
                self.s_1                    = self.s_0 * ( self.T_vis/self.T_rad )**(42./73.)

                # Eq. 13 -- the initial viscosity at s_0
                self.nu_0                        = ( self.alpha*gamma*physcon.k_B / (self.mu*physcon.m_p) ) * self.T_0/self.Omega_0
                if debug: print("debug:                nu_0 = %.2f m^2/kg" % ( self.nu_0 ) )

                # Eq. 14 -- initial gas mass accretion rate (absolute value is constant through the disk, actual value transitions from positive to negative at some radius)
                self.M_0_dot                = 3.0 * np.pi * self.Sigma_0 * self.nu_0
                if debug: print("debug:                M_0_dot = %.3e Msun/yr" % ( self.M_0_dot*yr/M_sun ) )

                # Eq. 23 -- applied just to get M_1_dot; general function defined later
                self.M_1_dot                = self.M_0_dot * ( self.s_1/self.s_0 )**((-19./10.)*(5./3.))
                if debug: print("debug:                M_1_dot = %.3e Msun/yr" % ( self.M_1_dot*yr/M_sun ) )

                # Eq. 47 -- the irradiated viscous timescale
                self.tau_rad                = 7.0*self.M_1 / ( 13.0*self.M_1_dot )
                if debug: print("debug:                tau_rad = %.3e yr" % (self.tau_rad/yr ) )

                # The inner evaporative zone
                # Eq. 59 -- note that both exponents in this constant depend on the assumed opacity, which Chambers (2009) took from Stepinski (1998) for Eq. 59
                self.Sigma_evap                = self.Sigma_0 * ( self.T_0 / self.T_vis )**(4./19.) * ( self.T_0 / self.T_e )**(14./19.)

        # Eq. 22 -- disk outer radius r_out is 's' in C09
        def get_rout(self, time ):
                temprout = self.s_0 * (self.get_M(time)/self.M_0)**(-2)
                if debug: print("debug:                r_out(Mdisk=%.3e Msun) = %.3e au" % ( self.get_M(time)/M_sun, temprout/auSI ) )
                return temprout

        # get the disk mass as f(t)
        def get_M(self, time ):
                if time < self.t_1:
                    # Eq. 25 -- get the disk mass as f(t)
                    tempM        = self.M_0 / ( 1.0 + time/self.tau_vis )**(3./16.)
                else:
                    # Eq. 46 -- get the disk mass as f(t)
                    tempM        = self.M_1 / ( 1.0 + (time-self.t_1)/self.tau_rad )**(7./13.)
                if debug: print("debug:                M_disk(t=%.3e yr) = %.3e Msun" % ( time/yr, tempM/M_sun ) )
                return tempM

        # get the absolute mass accretion rate through the disk
        def get_Mdot(self, time ):
                if time < self.t_1:
                    # Eq. 23 -- absolute mass accretion rate through the disk before t_1
                    tempMdot = self.M_0_dot * ( self.get_rout( time )/self.s_0 )**(-95./30.)
                else:
                    # Eq. 44 -- absolute mass accretion rate through the disk after t_1
                    tempMdot = self.M_1_dot * ( self.get_rout( time )/self.s_1 )**(-10./7.)
                if debug: print("debug:                M_dot(Mdisk=%.3e Msun) = %.3e Msun/yr" % ( self.get_M(time)/M_sun, tempMdot*yr/M_sun ) )        
                return tempMdot

        # Eq. 15, 33, 58 -- local mass surface density (gas)
        def get_Sigma(self, radius, time, status ):

                if status == 1:    # if status is 1 = pass, proceed to find the surface density
                        if self.get_r_evap(time) <= radius < self.get_r_t(time):
                                # Eq. 15 (= Eq. 33 for r<r_t)
                                tempSigma = self.Sigma_vis * ( self.get_Mdot( time )/self.M_0_dot )**(3./5.) * ( radius/self.s_0 )**(-3./5.)
                        elif radius < self.get_r_evap(time):
                                # Eq. 58
                                tempSigma = self.Sigma_evap * ( self.get_Mdot( time )/self.M_0_dot )**(17./19.) * ( radius/self.s_0 )**(-24./19.)
                        else:
                                # Eq. 33 for r>r_t (also top right of Table 2 in Alessi et al. 2016)
                                tempSigma = self.Sigma_rad * ( self.get_Mdot( time )/self.M_0_dot ) * ( radius/self.s_0 )**(-15./14.)
                elif status == 2:
                        tempSigma = 0
                #if debug: print("debug:                Sigma(r=%.3e au, t=%.3e yr) = None" % ( radius/auSI, time/yr ) )
                if debug: print("debug:                Sigma(r=%.3e au, t=%.3e yr) = %s kg/m^2" % ( radius/auSI, time/yr, str(tempSigma) ) )
                return tempSigma
        

         #Eq. 38 -- the current transition radius from the irradiated to the viscous regime
        
        def get_r_t(self, time):
                temprt = self.s_0 * ( self.Sigma_rad/self.Sigma_vis )**(70./33.) * ( self.get_Mdot(time)/self.M_0_dot )**(28./33.)
                if debug: print("debug:                r_t(time=%.3e yr) = %.3e au" % ( time/yr, temprt/auSI ) )
                return temprt

        # Eq. 60 -- the outer radius of the inner evaporative zone
        def get_r_evap(self, time):
                tempre = self.s_0 * ( self.Sigma_evap/self.Sigma_vis )**(95./63.) * ( self.get_Mdot(time)/self.M_0_dot )**(4./9.)
                if debug: print("debug:                r_evap(time=%.3e yr) = %.3e au" % ( time/yr, tempre/auSI ) )
                return tempre

        # Eq. 8, 35 -- local midplane temperature
        def get_T(self, radius, time, status ):
                if status == 1:
                    if radius < self.get_r_evap( time ):
                        # Eq. .. (bottom left in Table 2 of Alessi et al. 2016)
                        tempT = ( self.T_0 * self.Sigma_0 / self.Sigma_evap ) * ( self.get_Mdot(time)/self.M_0_dot )**(2./19.) * ( radius/self.s_0 )**(-9./38.)
                    elif time < self.t_1:    # At time >= t_1, we're still in the purely viscous regime, so can use Eq. 8
                        # Eq. 8
                        tempT = self.T_vis * (self.get_Sigma( radius, time, status )/self.Sigma_0 )**(2./3.) * ( self.get_Omega(radius)/self.Omega_0 )**(1./3.)
                    else:
                        # Eq. 35
                        tempT = self.T_0 * ( self.get_Mdot(time)/self.M_0_dot ) * ( self.get_Sigma( radius, time, status )/self.Sigma_0 )**(-1) * ( radius/self.s_0 )**(-3./2.)
                elif status == 2:
                    tempT = np.nan
                if debug: print("debug:                T(r=%.3e au, t=%.3e yr) = %.3e K" % ( radius/auSI, time/yr, tempT ) )
                return tempT

        # Eq. 12 -- local viscosity
        def get_nu(self, radius, time, status):
                tempnu = self.nu_0 * ( self.get_T(radius,time, status )/self.T_0 ) * ( self.get_Omega(radius)/self.Omega_0 )**(-1)
                if debug: print("debug:                nu(r=%.3e au, t=%.3e yr) = %.2f m^2/kg" % ( radius/auSI, time/yr, tempnu ) )
                return tempnu

        # The local isothermal pressure scaleheight at (radius,temperature)
        def get_scaleheight(self, radius, temperature ):
            return self.get_cs( temperature ) / self.get_Omega( radius )

        # The density at the midplane at (radius,temperature,time)
        def get_rho_midplane(self, radius, temperature, time, status ):
            rho0 = np.sqrt(2.0*np.pi)**(-1) * self.get_Sigma( radius, time, status ) / self.get_scaleheight( radius, temperature )
            return rho0

        def tstruct(self, time):
                return [ self.get_T(rrr, time, self.validate(radius=rrr, time=float(time)) ) for rrr in self.rstruct() ]

        ### This is effectively the chemical network part i.e., it takes a disk structure and outputs the gas- and solid-phase abundance of each element
        #print("TODO: Chemistry part needs to return the full composition of ices and gas, not just the total elemental abundance in either phase.")
        #print("TODO: Add a way to add more phases easily, in particular we may want a small and large dust grain population, or in fact any number of dust bins. Good for transport processes later. Though this could be factored into the dust surface area calculation, which would then have to communicate with the dust population evolution, but only return one dust 'area density' or something per timestep per location.")

        def composition(self, time, specieslist, loc=np.array([False])):

                if not np.all(loc):
                        Ncells      = len(self.rstruct())
                else:
                        Ncells      = len(loc)
                Cgas                = np.zeros( Ncells )
                Cice                = np.zeros( Ncells )
                Ngas                = np.zeros( Ncells )
                Nice                = np.zeros( Ncells )
                Ogas                = np.zeros( Ncells )
                Oice                = np.zeros( Ncells )
                Pgas                = np.zeros( Ncells )
                Pice                = np.zeros( Ncells )
                Sgas                = np.zeros( Ncells )
                Sice                = np.zeros( Ncells )
                temptemp        = self.tstruct(time)

                if loc[0] == False:
                        for ii in range( len( self.rstruct() ) ):
                                if self.validate(radius=self.rstruct()[ii], time=time) !=2 :
                                        for key in specieslist.keys():
                                                if temptemp[ii] > specieslist[key].Tsub:
                                                        Cgas[ii]        += specieslist[key].Cabun
                                                        Ngas[ii]        += specieslist[key].Nabun
                                                        Ogas[ii]        += specieslist[key].Oabun
                                                        Pgas[ii]        += specieslist[key].Pabun
                                                        Sgas[ii]        += specieslist[key].Sabun
                                                elif temptemp[ii] <= specieslist[key].Tsub:
                                                        Cice[ii]        += specieslist[key].Cabun
                                                        Nice[ii]        += specieslist[key].Nabun
                                                        Oice[ii]        += specieslist[key].Oabun
                                                        Pice[ii]        += specieslist[key].Pabun
                                                        Sice[ii]        += specieslist[key].Sabun
                        return {'C/H_gas':Cgas, 'O/H_gas':Ogas, 'N/H_gas':Ngas, 'S/H_gas':Sgas,
                                'P/H_gas':Pgas, 'C/H_ice':Cice, 'O/H_ice':Oice, 'N/H_ice':Nice,
                                'S/H_ice':Sice, 'P/H_ice':Pice}
                else :
                        for ii in range( len(loc) ):
                                if self.validate(radius=loc[ii], time=time) !=2 :
                                        for key in specieslist.keys():
                                                temp = self.get_T(loc[ii], time, self.validate(radius=loc[ii], time=float(time)) )
                                                if temp > specieslist[key].Tsub:
                                                        Cgas[ii]        += specieslist[key].Cabun
                                                        Ngas[ii]        += specieslist[key].Nabun
                                                        Ogas[ii]        += specieslist[key].Oabun
                                                        Pgas[ii]        += specieslist[key].Pabun
                                                        Sgas[ii]        += specieslist[key].Sabun
                                                elif temp <= specieslist[key].Tsub:
                                                        Cice[ii]        += specieslist[key].Cabun
                                                        Nice[ii]        += specieslist[key].Nabun
                                                        Oice[ii]        += specieslist[key].Oabun
                                                        Pice[ii]        += specieslist[key].Pabun
                                                        Sice[ii]        += specieslist[key].Sabun
                        return {'C/H_gas':Cgas, 'O/H_gas':Ogas, 'N/H_gas':Ngas, 'S/H_gas':Sgas,
                                'P/H_gas':Pgas, 'C/H_ice':Cice, 'O/H_ice':Oice, 'N/H_ice':Nice,
                                'S/H_ice':Sice, 'P/H_ice':Pice}

class DiskPlanetforming:
        """
        A class for a planet-forming disk, subclassed from the provided disk_subclass.
        """
        def __init__(self, disk_subclass = disk_viscirr, **kwargs):
                # Dynamically set the parent class (disk_subclass) and initialize it
                self.__class__ = type(self.__class__.__name__, (disk_subclass, DiskPlanetforming), {})
                super(self.__class__, self).__init__(**kwargs)  
        
# =============================================================================
#         SOLID SURFACE DENSITIES
# =============================================================================

        def get_Sigsolid(self, radius, time, status ):
                if status == 1:    # if status is 1 = pass, proceed to find the surface density of solids
                        tempSigmasol = self.get_Sigma(radius, time, status) * self.dtgr 
                elif status == 2:
                        tempSigmasol = 0
                return tempSigmasol

        def get_peb_flux(self, time, radius, status):
                vkep = self.get_Omega(radius) * np.pi * 2 * radius

                st          = self.stokes_calculator(radius, time, status)
                rho_gas     = self.get_Sigma(radius, time, status) / (np.sqrt(2*np.pi)*self.get_scaleheight(radius, self.get_T(radius, time, status)))
                p_gas       = rho_gas * self.get_cs(self.get_T(radius, time, status))**2
                rInt        = np.zeros(np.size(self.rstruct())+ 1)
                rInt[1:-1]  = 0.5 * (self.rstruct[1:]+self.rstruct[:-1])
                rInt[-1]    = 1.5 * self.rstruct[-1] - 0.5 * self.rstruct[-2]
                dr          = rInt[1:] - rInt[:-1]
                pgInt       = np.interp(rInt, self.rstruct(), p_gas )                                             # gas pressure at the interfaces, peble predictor
                #eta_pp      = (pgInt[1:]-pgInt[:-1])/dr[:] /(2.*rho_gas*self.get_Omega(radius)**2.*self.rstruct())
                eta_c       = self.get_cs(self.get_T(radius, time, status), gamma = 7./5.)/ vkep 
                peb_drift_c = (2 * eta_c * vkep * st)/(1 * st**2)
                #peb_drift   = (2 * abs(eta) * self.get_Omega(radius) * self.rstruct() * st) / (1 + st**2)

                return peb_drift_c

        def get_pebsig(self, radius, time, status): # pebble surface density using dust & gas ratio and percentage of solid surface desnity that is made of pebbles
                '''
                Calculates pebble surface density at given radius and time for specified Stokes number.
                If stokes number > 1e-3, then it's an accretable pebble. Otherwise, set Stokes number
                to 0 so that it doesn't contribute to pebble surface density.
                '''
                # if status == 1:
                #         pebstokes = stokes if stokes > 1e-3 else 0
                #         Sigsolid = self.get_Sigsolid(radius, time, status=1)
                #         pebsig = 2 * Sigsolid * pebstokes / (self.rhos * np.pi)
                #         return pebsig

                if status == 1:    # if status is 1 = pass, proceed to find the surface density of pebbles
                        temppebsig = self.get_Sigsolid(radius, time, status) * self.pebr
                elif status == 2:
                        temppebsig = 0
                return temppebsig
        
        def get_disk_plan(self, radius, status): # not dependent on time, assumes disk of planetesimals forms at t=0 as fraction of dust surface density
                if (status == 1):# and (radius < self.s_0):    
                        time        = 1 * yr
                        if radius < self.s_0:
                                tempplansig = self.get_Sigsolid(radius, time, status) * self.plnr
                        elif radius > self.s_0:
                                tempplansig = 0
                else:
                        tempplansig = 0
                return tempplansig

        def get_init_grid(self, status): # makes planetesimal surface density constant across radial grid cells
                rad_grid = np.array(self.rstruct())
                rad_grid_centres = 0.5 * (rad_grid[1:] + rad_grid[:-1])
                planetesimal_surface_density = [self.get_disk_plan(radius = r, status = status) for r in rad_grid_centres]
                grid_sig = np.vstack((rad_grid[:-1], planetesimal_surface_density)).T
                return grid_sig

        def surface_density_profile(self, radius, status):
                return self.get_disk_plan(radius, status)

        def int_plan_surf(self, r_in, r_out, status):
                def integrand(radius):
                        return self.surface_density_profile(radius, status)
        
                surf_int = integrate(integrand, r_in, r_out)
                return surf_int

        def get_M_numerical(self, disk, time_array, radius_array):
                return np.array([np.sum([(radius_array[i+1] - radius_array[i]) * 2 * np.pi * ((radius_array[i] + radius_array[i+1]) / 2) * disk.get_Sigma(time, (radius_array[i] + radius_array[i+1]) / 2) for i in range(len(radius_array)-1)]) for time in time_array])/self.M_star

# =============================================================================
#           STOKES NUMBERS & GRAIN SIZES
# =============================================================================

        def get_St_turb(self, temperature): # Eq 4 (chambers 2018 popsyn), stokes number due to turbulence
            St =  (self.vfrag**2) / (3 * self.alpha * (self.get_cs(temperature, gamma = 7/5)**2))
            return St
        
        def get_St_drft(self, radius, temperature): # Eq 4 (chambers 2018 popsyn), stokes number due to drift
                vkep = self.get_Omega(radius) * np.pi * 2 * radius
                eta = (self.get_cs(temperature)/vkep)**2
                St =  ( self.vfrag ) / ( eta * vkep)

                return St
   
        def get_St_drift_barrier(self, status): # Function to calculate the Stokes number at the radial drift barrier. Depends on dust-to-gas ratio, gas pressure gradient, and a fudge factor from Drazkowska et al. 2021.

                if status == 1:
                        eta = 1e-3 # approximation for gas pressure gradient (found after Eq. 9 Krijt et al. 2018 [so we can blame seb])
                        St = self.dtgr / (30 * eta)
                elif status == 2:
                        St = 0
                return St

        def get_st_ini(self, time, radius, status):
                if status == 1:
                        sigma = self.get_Sigma(radius, time, status) # surface density of gas
                        pebble_size = 1e-6                              # m
                        rho_solid = 1250                                # internal density of solids in SI units, using number from Drazkowska + 2023
                        Z0 = self.dtgr                                  # dust to gas ratio   
                        # calculating stokes numbers [generic equation from e.g: Eq. (2) from Birnstiel+2012]
                        St = (pebble_size * rho_solid/sigma) * (np.pi/2)
                elif status == 2:
                        St = 0
                return St

        def stokes_calculator(self, radius, time, st, status):
                
                Z0 = self.dtgr                                  # dust to gas ratio   
                omega = self.get_Omega(radius)

                # growth timescales based on Draskowska+21, pebble predictor
                # Eq. (7) Drazkowska+21
                tgrowth  = Z0 * (1/omega) * (self.alpha/1e-4)**(-1/3) * (radius/auSI)**(1/3)

                growth_factor = np.exp((time/yr)/tgrowth) 

                stokes = st * growth_factor

                # comparing stokes number against turbulence- and drift-induced fragmentation barriers
                st_drift_frag = self.get_St_drft(radius, self.get_T(radius, time, status))
                st_turb_frag = self.get_St_turb(self.get_T(radius, time, status))

                # comparing fragmentation barriers against stokes number
                stokes = np.minimum(np.minimum(st_drift_frag, st_turb_frag), stokes)

                return stokes

        def get_grainsize(self, st, sigma_peb, status): # convert stokes number to grain size
                rho_solid = 1250
                if status == 1:
                        a = ((2 * st * sigma_peb)/ self.dtgr)/(rho_solid * np.pi)
                elif status == 2:
                        a = None
                return a

# =============================================================================
#           OTHER
# =============================================================================
        
        def get_hr(self, mass, radius, status): # m, hill radius, armitage 2020
                if status == 1:
                        hr = radius * ( mass / (3 * self.M_star))**(1/3)
                elif status == 2:
                        hr = 0
                return hr

        def get_mr(self, mass): # m, mass radius relation, bashi 2017
            if mass < (124 * M_e):
                        r     = 1.286e-7 * mass**(0.55)
            else:
                        r     = 4.157e7 * mass**(0.01)
            return r

        def get_br(self, mass, temperature): # m, bondi radius
            br  = (2 * physcon.G * mass)/(self.get_cs(temperature, gamma = 7/5)**2)
            return br
        
        def get_fg(self, mass, r_plan, vrel): # dimensionless, grav focussing factor
            fg  = 1 + ( ( (2 * physcon.G * mass) / r_plan)**0.5 / vrel )**2
            return fg

        def get_eta(self, temperature, vkep): # dimensionless
            eta  = (self.get_cs(temperature, gamma = 7/5)/vkep)**2
            return eta

        def get_hpeb(self, radius, temperature, st): # m, scale height of pebble disk
            hpeb  = (self.get_scaleheight(radius, temperature) * ( self.alpha / (self.alpha + st) ))**(1/2)
            return hpeb

# =============================================================================
#           ISOLATION AND GAP OPENING MASSES
# =============================================================================

        def get_iso_peb(self, radius, temperature, status): # Eq 23 brugger et al 'pebbles vs planetesimals', pebble isolation mass
                if status == 1:
                        iso =  20 * ( (self.get_scaleheight(radius, temperature)/radius) /0.05 )**3 * M_e
                elif status == 2:
                        iso = 0
                return iso

        def get_mgap(self, time, radius, temperature, status): # kg, gap opening mass, chambers 2018 eq 12, 16-19
            vkep              = self.get_Omega(radius) * 2 * np.pi * radius
            H_gas             = self.get_scaleheight(radius, temperature)
            
            Q                 = (self.get_cs(temperature, gamma = 7/5) * vkep) / (np.pi * radius * physcon.G * self.get_Sigma(radius, time, status) )
            M1                = (2 * radius * self.get_cs(temperature, gamma = 7/5)**3) / (3 * physcon.G * vkep)
            
            M_vis             = M1 * ( (self.alpha * radius) / (0.043 * H_gas) )**(0.5)
            M_inv             = M1 * np.min([ 5.2 * Q**(-5/7) , 3.8 * (( Q * radius )/( H_gas ))**(-5/13) ])
            
            if status == 1:
                        mgap  = np.max([M_vis, M_inv])
                                
            elif status == 2:
                        mgap = np.nan
            return mgap

# =============================================================================
#       SOLID ACCRETION/ GAS ACCRETION/ ENVELOPE CONTRACTION
# =============================================================================

        def get_envcon(self, mcore, menv, temperature, status): # envelope contraction bitsch et al 2019 eq 7
            if status == 1:
                        f                  = 0.2
                        k_env              = 0.05       # cm^2/g
                        rho_c              = 5.51       # g/cm^3
                        con                = 0.000175 * f**(-2) * (k_env/1)**(-1) * (rho_c/5.5)**(-1/6) * (mcore/M_e)**(11/3) * (menv/M_e)**(-1) * (temperature/81)**(-1/2) * (M_e/(1e6*yr))

            elif status == 2:
                        con = np.nan
            return con

        def get_gasacc(self, time, radius, mass, temperature, status): # kg/s, bitsch et al 2019, eq 8 , 9 
            r_h                = self.get_hr(mass, radius, status)
            H                  = self.get_scaleheight(radius, temperature)   
            if (r_h/H) < 0.3:
                        gasacc_low   = 0.83 * self.get_Omega(radius) * self.get_Sigma(radius, time, status) * H**2 * (r_h/H)**(9/2)
                        gasacc       = gasacc_low 
            elif (r_h/H) >= 0.3:
                        gasacc_high  = 0.14 * self.get_Omega(radius) * self.get_Sigma(radius, time, status) * H**2
                        gasacc       = gasacc_high 
            gas_lim                  = min(gasacc, 0.8*self.get_Mdot(time))
            return gas_lim

        def get_pebacc(self, rcap_peb, radius, vrel, temperature, st, sigma): # kg/s, pebble accretion, chambers 2018
                hpeb               = self.get_hpeb(radius, temperature, st)
                sig_peb            = sigma
                if st >= 1e-4:
                        if (rcap_peb) > hpeb: 
                                pebacc   = 2 * rcap_peb * vrel * sig_peb
                        elif (rcap_peb) < hpeb:
                                pebacc   = (np.pi * rcap_peb**2 * vrel * sig_peb) / (2 * hpeb)
                else:   
                        # print(f'NEGATIVE STOKES {st}')
                        pebacc = 0
                return pebacc

        def get_plnacc(self, sig_plan, radius, rcap_plan, fg): # kg/s, planetesimal accretion, mordasini 2018
            fkep      = self.get_Omega(radius)
            planacc   = fkep * sig_plan * (rcap_plan**2) * fg
            return planacc

# =============================================================================
#           MIGRATION
# =============================================================================

        def get_migI(self, time, radius, mass, temperature, status): # type 1 migration function - e. gladwin from tanaka et al 2002 eqs
            tt_1         = ( 1.160 + (self.alpha * 2.828) )
            tt_2         = ( ( mass / self.M_star ) * ( ( radius * self.get_Omega(radius) ) / self.get_cs(temperature, gamma = 7/5) ) )**2
            tt_3         = radius **4 * self.get_Omega(radius)**2 * self.get_Sigma(radius, time, status)
            tidal_torque = tt_1 * tt_2 * tt_3
            l_p          = mass * (physcon.G * self.M_star * radius)**(1/2) 
            if status == 1:
                        mig          = -2 * radius * (tidal_torque/l_p)
            elif status == 2:
                        mig = np.nan
            return mig

        def get_migII(self, time, radius, mass, status): # type 2 migration function - e. gladwin from scardoni et al 2019
            if status == 1:
                        B = ((4 * np.pi * radius**2 * self.get_Sigma(radius, time, status) ) / mass )
                        mig = - self.get_gasv(time, radius, status) * ( B / (B+1) ) 
            elif status == 2:
                        mig = np.nan
            return mig

        def get_mig(self, time, radius, mass, temperature, status): # type 1 and 2 migration rates from tanaka 2002 and scardoni 2019 respectively, boundary defined by alibert 2005
            m_gap               = self.get_mgap(time, radius, temperature, status)
            if mass < (0.5 * m_gap):
                        mig = self.get_migI(time, radius, mass, temperature, status)
            else:
                        mig = self.get_migII(time, radius, mass, status)                   
            return mig

        def get_migk(self, time, radius, mass, temperature, status): # combined migration from kanagawa et al 2018
            v_k                 = self.get_Omega(radius)
            h_s                 = self.get_scaleheight(radius, temperature)/radius
            m_gap               = self.get_mgap(time, radius, temperature, status)
            sig_gas             = self.get_Sigma(radius, time, status)
            f_1                 = 1
            f_2                 = -1
            f_s                 = 1 / (1 + (mass/m_gap)**4 )
            f_tot               = (f_1*f_s) + (f_2 * ( 1-f_s )) * (1/(mass/m_gap)**2)
            if status == 1:
                        mig = f_tot * (mass/self.M_star) * ((sig_gas * radius**2)/self.M_star) * h_s**(-2) * v_k
            elif status == 2:
                        mig = np.nan
            return mig

        def get_migjc(self, time, radius, mass, temperature, x, status): # modified type one migration from chambers 2018 popsyn eq15
            vkep                    = self.get_Omega(radius) * 2 * np.pi * radius
            m_gap                   = self.get_mgap(time, radius, temperature, status)
            if status == 1:
                        if mass < (0.5 * m_gap):
                                m_migeff    = mass
                        elif mass > (0.5* self.get_mgap(time, radius, temperature, status)):
                                m_migeff    = (0.5 * m_gap) * (mass / ( 0.5 * m_gap) )**(x)
                        
                        mig         = -3.8 * vkep * (m_migeff/ self.M_star) * ( (self.get_Sigma(radius, time, status) * radius**2 ) / self.M_star) * (vkep/self.get_cs(temperature, gamma = 7/5))**2
            elif status == 2:
                        mig = np.nan
            return mig
        
        def get_PDM(self, radius, sigma, mass, bh_pdm, mass_horseshoe, status):
                if status == 1:
                        mass_disk = np.pi * sigma * (radius ** 2) # kg
                        period_T = ((radius)/auSI) ** (3/2) # keps big 3! yrs
                        hill_radius = self.get_hr(mass, radius, status) # m
                        hill_factor =  hill_radius / radius # dimensionless
                        mass_encounter = 5 * hill_factor * mass_disk # kg
                        fid_rate = (((2 * mass_disk * (radius/auSI)) / (self.M_star * period_T))) # AU/yr
                        one_scatter_lim = ((bh_pdm * hill_factor)**2) * 3*(radius/auSI)/ (2*period_T) #AU/yr

                        if mass_horseshoe/(2 * mass_encounter) > 1 : # if mass of horseshoe exceeds mass of encounter
                                factor = 1 + (mass_horseshoe/(2*mass_encounter))**2 # dimensionless
                                dadt = fid_rate / factor # m/s
                        # elif hill_eccentricity > 1:
                        #         factor = 1 + (eccentricity_rms/(3*hill_factor)**3)
                        #         dadt = fid_rate / factor
                        
                        else:
                                if (mass/mass_encounter < 1) and (radius > (0.02)):
                                        if fid_rate > one_scatter_lim:
                                                factor = 1
                                                dadt = fid_rate * factor
                                        else:
                                                factor = 0.5
                                                dadt = fid_rate * factor
                                elif (1 < mass/mass_encounter < 10) and (radius > (0.02)):
                                        factor = 1 + ((1/5)*((mass/mass_encounter)**3))
                                        dadt = fid_rate / factor
                                else:
                                        dadt = 0 # m/s
                elif status == 2:
                        dadt = np.nan # m/s
                
                return dadt

# =============================================================================
#       SURFACE DENSITY CHANGES
# =============================================================================

        def get_vr(self, time, radius, status): # radial drift velocity of pebbles chambers 2018 eq 6
            if status == 1:
                        vkep = self.get_Omega(radius) * 2 * np.pi * radius
                        eta = self.get_eta(self.get_T(time, radius, status), vkep)
                        St = 1e-2 #self.stokes_calculator(radius, time, status)
                        v_r = (2 * eta * vkep * St) / (1 + St**2)
            elif status == 2:
                        v_r = np.nan
            return v_r
        
        def get_plan_grid(self, input_s, status):
                rad_grid = np.array(self.rstruct())
                r_grid_centres = 0.5*(rad_grid[1:] + rad_grid[:-1])
                planetesimal_surface_density = [self.get_disk_plan(radius=r, status=1) for r in r_grid_centres]
                grid_sig = np.vstack((rad_grid[:-1], planetesimal_surface_density)).T
                if status == 1:
                        bindex = np.digitize(input_s, grid_sig[:,0])-1
                        if bindex < 0 or bindex >= len(rad_grid):
                                sys.exit("!! Radius r=%.3e au out of bounds" % (input_s/auSI) )
                                sigma = np.NaN
                        else:
                                sigma = grid_sig[bindex,1]
                elif status == 2:
                        sigma = np.NaN
                return grid_sig, sigma

        def update_plan_grid(self, input_s, planetesimal_grid, B, m_dot, mass, dt, status):
                grid_sig                = planetesimal_grid
                hillrad                 = self.get_hr(mass, input_s, status)
                fz_inner                = input_s - ((0.5 * B) * hillrad) 
                fz_outer                = input_s + ((0.5 * B) * hillrad)
                grid_edges              = grid_sig[:,0]
                feeding_indicies        = np.array(np.where((grid_edges >= fz_inner) & (grid_edges <= fz_outer))).T
                sig_con                 = - 3 * self.M_star**(1/3) / (6 * np.pi * (input_s**2) * B)
                sig_dec                 = sig_con * mass**(-1/3) * m_dot * dt
                bindex                  = np.digitize(input_s, grid_edges)-1
                # if bindex < 0 or bindex >= len(planetesimal_grid[:,0]):
                #         sys.exit("!! Radius r=%.3e au out of bounds" % (input_s/auSI) )
                for i in feeding_indicies:
                        grid_sig[i,1] += sig_dec
                        grid_sig[i,1] = max(grid_sig[i,1], 0)
                sigma_mod = grid_sig[bindex,1]
                return grid_sig, sigma_mod
