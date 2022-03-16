
##  Purpose:Water Balance Model (1D homogenous soil profile)
##  calc.soil_plant_water_balance seeks to reproduce this work, as well as Ingrid's
##  One of the functions was copied in the tests to be used as a test against it's calc counterpart
##  Authors:Eunjin Han & Amor VM Ines
##  Collab'tors:	Walter Baethgen
##  Institute:IRI-Columbia University, NY
##  Revision dates:
##  6/7/2016 => All grids computations are here (not on ArcGIS, which takes too much time)
##             Then, final output will be saved as *.asc and visualized on ArcGIS
##  5/16/2016, EJ started following Amor's Fortran program - A polygon-based Water Balance Model (1D homogenous soil profile)
##    !A L G O R I T H M (Amor's SWB Model) => EJ modified slightly:
##    !Read PARAMETER FILE
##    !	-control for crop management
##    !		-sowing date
##    !		-Kc values for ini, mid, late, length of devt stages
##    !	-control for soil property database
##    !	-control for ETref method - later capability?
##    !Read POLYGON FILE
##    !Iterate across RECORDS/POLYGONS
##    !	- from attibute table (simplified)
##    !CHECK CENTROID LAT, LONG, SOIL
##    !	- from attribute table (simplified)
##    !TRIANGULATE NEAREST STATION
##    !	-Minimize D=sqrt[(latp(k)-lats(j))^2+(lonp(k)-lons(j))^2]; k/=j
##    !DERIVE SOIL PROPERTIES FOR THAT RECORD
##    !
##    !START Calculations:
##    !DEVELOP MODULES FOR CACULATIONS
##    !ETref - Hargreaves
##    !	- calculate Ra - needs lat-lon, Julian Day
##    !		:Rs-Ra MODULE
##    !Kc - linear interpolation
##    !Ks - trapezoidal rule
##    !API-Peff
##    !
##    !Water balance calculation (1D homogenous soil profile):
##    !S(i)-S(i-1)=Peff(i)-[ETcrop_red(i)+Dr(i)], use Ines-Baethgen-Cousin Algorithm
##    !S(i)+Dr(i)=S(i-1)+Peff(i)-ETcrop_red(i)
##    !W(i)=S(i-1)+Peff(i)-ETcrop_red(i); where W(i)=S(i)+Loss(i); where Loss(i)=Dr(i)
##    !
##    !IF W(i)>TAW; S(i)=TAW; Loss(i)=W(i)-S(i); this is deep vertical drainage (1-layer soil water balance)
##    !ELSEIF W(i)<TAW; S(i)=W(i)
##    !Continue the next day
##    !
##    !Summary Report
##    !A L G O R I T H M:

##import datetime    #to convert date to doy or vice versa
##import subprocess  #to run executable
##import shutil   #to remove a foler which is not empty
import os   #operating system
import numpy as np
#import matplotlib.pyplot as plt  #to create plots
import fnmatch   # Unix filename pattern matching => to remove PILI0*.WTD
import os.path
#from scipy.stats import rankdata #to make a rank to create yield exceedance curve
import math
import calendar  #for isleap function
import matplotlib.pyplot as plt  #to create plots
import time
from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime, timedelta
import csv

#===DEFINE FUNCTIONS=====================================
# Determine runoff and effective precipitation based on SCS curve number method (EJ (12/20/2019))
def Peffective_2D(PCP, CN):  # CN should be pre-defined based on land cover, hydrologic soil groups, and antecent soil moisture condition
    #PCP is 600 x 600 matrix
    #potential maximum retention after runoff begins
    S_int = 25400/CN - 254  #Need to updae this if CN is a map
    numerator = PCP - 0.2*S_int #0.2*S_int => initial abstractions
    numerator = np.multiply(numerator, numerator)  
    denominator = PCP + 0.8*S_int
    Runoff = np.divide(numerator, denominator)
    Runoff[PCP < 0.2*S_int] = 0
    Runoff[PCP <= 0] = 0
    Runoff[Runoff < 0] = 0

    Peff = PCP - Runoff
    return Peff, Runoff

# def Peffective(PCP, CN):  # CN should be pre-defined based on land cover, hydrologic soil groups, and antecent soil moisture condition
#     #potential maximum retention after runoff begins
#     S_int = 25400/CN - 254  #Need to updae this if CN is a map
#     temp1 = np.square(PCP - 0.2*S_int)  #0.2*S_int => initial abstractions
#     temp2 = PCP + 0.8*S_int
#     Runoff = np.divide(temp1, temp2)
#     Runoff[PCP < 0.2*S_int] = 0
#     Runoff[PCP <= 0] = 0
#     Runoff[Runoff < 0] = 0

#     Peff=PCP - Runoff
#     return Peff, Runoff


def ETref_PM_2D(lat,doy,mselev,SRad,Tmaximum,Tminimum):    #FAO Penman-Monteith (EJ: May 17, 2016)
    #All equations are from "Allen, Richard G., et al. "FAO Irrigation and drainage paper No. 56." Rome: Food and Agriculture Organization of the United Nations (1998): 26-40.
    #mask out input based on DEM coverage
    Tmaximum[mselev < 0] = np.nan
    Tminimum[mselev < 0] = np.nan
    SRad[mselev < 0] = np.nan

    #compute avg temp
    # Tavg=0.5*(Tminimum + Tmaximum)  #celsius
    Tavg = np.add(Tminimum, Tmaximum)
    Tavg = np.multiply(Tavg, 0.5)
    
    #1) compute Ra(extraterrestrial radiation)
    #1-1) convert latitude from decimal degrees to raidans (FAO eq#22)
    lat_rad = (math.pi /180)*lat  #=> 2D matrix
    #1-2) calculate the inverse relative distance Earth-Sun, d_r (FAO eq#23)
    d_r = 1+0.033*np.cos(2*math.pi*doy/365)  #=> constant . fn(day)
    #1-3) calculate solar declimation, s_d (FAO eq#24)
    s_d = 0.409*np.sin(2*math.pi*doy/365-1.39) #=> constant . fn(day)
    #1-4) calculate sunset hour angle (FAO eq#25)
    sha = np.arccos(-1 * np.multiply(np.tan(lat_rad), np.tan(s_d)))  #=>2D matrix . fn(lat, day)
    #1-5) calculate Ra(extraterrestrial radiation) (FAO eq#21) unit: MJ/m2/day
    #solar constant = 0.0820 MJm^(-2) min^(-1)
    temp = np.multiply(sha, np.sin(lat_rad))
    temp = np.multiply(temp, np.sin(s_d))
    temp2 = np.multiply(np.cos(lat_rad), np.cos(s_d))
    temp2 = np.multiply(temp2, np.sin(sha))
    temp3 = np.add(temp, temp2)
    Ra = np.multiply(24*60*0.0820*d_r/math.pi, temp3)
    
    #2) CALCULATE PARAMETERS FOR FAO P-M EQUTIONS
    #2-1) calculate the slope of saturation vapor pressure curve, delta_sl at air temperature (eq #13)
    temp = np.multiply(17.27, Tavg)
    temp = np.divide(temp, Tavg+237.3)
    numerator = np.exp(temp)
    numerator = np.multiply(0.6108, numerator)
    numerator = np.multiply(4098, numerator)
    denominator = np.power(Tavg+237.3, 2)
    #delta_sl=4098*(0.6108*math.exp(17.27*Tavg/(Tavg+237.3)))/(math.pow((Tavg+237.3), 2)) #unit: kPa/deg C
    delta_sl = np.divide(numerator, denominator) #=>2D matrix 
    #2-2) calculate the atmospheric pressure, p (eq #7)
    #mselev= elevation above sea level [m]
    numerator = np.multiply(0.0065, mselev)
    numerator = np.subtract(293, numerator)
    temp = np.divide(numerator, 293)
    temp = np.power(temp, 5.26)
    atmP = np.multiply(101.3, temp)  #=>2D matrix 
    #p = 101.3*math.pow(((293-0.0065*mselev)/293),5.26)
    #2-3) calculate the psychrometric constant, gamma (eq #8)
    gamma = np.multiply(0.665*0.001,atmP)  #=>2D matrix 
    #gamma = 0.665*math.pow(10,-3)*p

    #2-4) Assume wind speed is 2 m/s and calculate 3 factors which will be used in the final P-M equation (eq#6)
    u2=2.0  #wind speed at 2m height [m/s]
    # f1=delta_sl/(delta_sl+gamma*(1 + 0.34*u2))
    temp = np.multiply(0.34, u2)
    temp = np.add(1,temp)
    temp = np.multiply(gamma,temp)
    temp = np.add(delta_sl, temp)
    f1 = np.divide(delta_sl, temp) 
    # f2=gamma/(delta_sl+gamma*(1 + 0.34 * u2))
    f2 = np.divide(gamma,temp)
    # f3=900*u2/(Tavg+273)
    temp = np.add(Tavg, 273)
    temp = np.divide(900,temp)
    f3 = np.multiply(temp,u2)

    #2-5) Estimate the vapour pressure deficit (Es-Ea) from tmax and tmin because relative humidity is not available
    #assume Tdew ~ Tmin (usually OK for humid regions but not always true for arid regions)
    Tdew = Tminimum
    #calcuate actual vapour pressure derived from Tdw (eq#14) [kPa]
    # Ea=0.6108*math.exp((17.27*Tdew)/(Tdew+237.3))
    temp = np.multiply(17.27, Tdew)
    temp2 = np.divide(temp, np.add(Tdew,237.3))
    Ea = np.multiply(0.6108, np.exp(temp2))

    #calcuate MEAN saturation vapour pressure from Tmin and Tmax (eq #11, #12)
    #- a) calcuate saturation vap pressure at tmax (eq#11)
    # Emax=0.6108*math.exp((17.27*Tmaximum)/(Tmaximum+237.3))
    numerator = np.multiply(17.27, Tmaximum)
    denominator = np.add(Tmaximum, 237.3)
    temp = np.exp(np.divide(numerator, denominator))
    Emax = np.multiply(0.6108, temp)  #=>2D matrix 
    #- b) calcuate saturation vap pressure at tmin (eq#11)   
    # Emin=0.6108*math.exp((17.27*Tminimum)/(Tminimum+237.3))
    numerator = np.multiply(17.27, Tminimum)
    denominator = np.add(Tminimum, 237.3)
    temp = np.exp(np.divide(numerator, denominator))
    Emin = np.multiply(0.6108, temp)  #=>2D matrix 
    #- c) calculate MEAN saturation vapour pressure (eq#12)
    # Es=(Emax + Emin)*0.5
    temp = np.add(Emax, Emin)
    Es = np.divide(temp, 2) #=>2D matrix 
    #- d) Estimate the vapour pressure deficit (Es-Ea)
    # VPD = Es-Ea
    VPD = np.subtract(Es, Ea)   #=>2D matrix 

##    #3) Estimate solar radiation derived from air temperature difference
    #-a) clear-sky solar radiation (Rso) (eq #37)
    # Rso=(0.75 + 2*math.pow(10,-5)*mselev)*Ra
    temp = np.multiply(2,10**-5)   #np.power(10,-5) does not work
    temp = np.multiply(temp, mselev)
    temp = np.add(0.75, temp)
    Rso = np.multiply(temp, Ra) # Ra(extraterrestrial radiation)
##    #-b) Krs: adjustment coefficient (0.16-0.19)=> for interior, Krs~0.16, for costal location, Krs~0.19
##    #       we assume Krs ~ 0.175 as an average value for all locations
##    Krs=0.175  
##    #-c) calcuate Rs (eq #50)
##    Rs=Krs*Ra* math.sqrt(Tmaximum-Tminimum)
    
    #use observed Rs [MJ/m2/day]
    Rs=SRad
    
#     if Rs > Rso:  #Rs predicted by eq#50 should be limited to <= Rso
#         Rs=Rso  #FAO p.61 Q: Do we need this precaution??
# ##        tkMessageBox.showerror('Error in Rs', 'Rs is greater than Rso!')
# ##        os.system("pause")
    Rs = np.where(Rs > Rso, Rso, Rs)  #Where True, yield x, otherwise yield y.
        
    #4)Calculate net solar or net shortwave radiation, Rns (eq #38)
    albedo = 0.23 #canopy reflection coeff => 0.23 for hypothetical grass reference crop 
    # Rns=(1-albedo)*Rs
    Rns = np.multiply((1-albedo), Rs)  #=>2D matrix 

    #5) Calcualate net longwave radiation(eq #39) - based on Stefan-Boltzmann law
    # sigma= 4.903*np.power(10,-9) #Stefan-Boltzmann constatn [MJ K^(-4) /m2/day]
    sigma= 4.903*(10**-9) #Stefan-Boltzmann constatn [MJ K^(-4) /m2/day]
    temp_a = (np.power((Tmaximum+273.16),4)+np.power((Tminimum+273.16),4))*0.5
    # temp_b = 0.34 - 0.14 * np.sqrt(Ea) 
    temp_b = np.multiply(0.14, np.sqrt(Ea))  #Ea = actual vapour pressure
    temp_b = np.subtract(0.34, temp_b)
    # temp_c = 1.35*Rs/Rso-0.35
    temp_c = np.divide(Rs,Rso) #clear-sky solar radiation (Rso) , observed Rs
    temp_c = np.multiply(1.35,temp_c)
    temp_c = np.subtract(temp_c,0.35)
    # Rnl=sigma*temp_a*temp_b*temp_c  #[MJ/m2/day]
    Rnl = np.multiply(sigma,temp_a)
    Rnl = np.multiply(Rnl,temp_b)
    Rnl = np.multiply(Rnl,temp_c)  #=>2D matrix 

    #6) Calculate net solar radiation, Rn (eq #40)
    # Rn = Rns - Rnl
    Rn = np.subtract(Rns, Rnl)

    #7) ===COMPUTE FAO P-M ET_0======================
    #from eq#42, assume the soi lheat flux, G =0
    # ETref = 0.408*Rn*f1 + f2*f3*VPD
    temp = np.multiply(0.408, Rn)
    temp = np.multiply(temp, f1)
    temp2 = np.multiply(f2, f3)
    temp2 = np.multiply(temp2, VPD)  #vapour pressure deficit (Es-Ea)
    ETref = np.add(temp, temp2)

    #mask out
    ETref[np.isnan(Tmaximum)] = np.nan
    ETref[np.isnan(SRad)] = np.nan
    return ETref


def ETref_Harg_2D(lat,doy,mselev, Tmaximum,Tminimum):    #Method 1: Hargreaves
    #All equations are from "Allen, Richard G., et al. "FAO Irrigation and drainage paper No. 56." Rome: Food and Agriculture Organization of the United Nations (1998): 26-40.
    #mask out input based on DEM coverage
    Tmaximum[mselev < 0] = np.nan
    Tminimum[mselev < 0] = np.nan

    #compute avg temp
    # Tavg=0.5*(Tminimum + Tmaximum)  #celsius
    Tavg = np.add(Tminimum, Tmaximum)
    Tavg = np.multiply(Tavg, 0.5)
##    what if Tavg< 0?
    
    #1) compute Ra(extraterrestrial radiation)
    #1-1) convert latitude from decimal degrees to raidans (FAO eq#22)
    lat_rad = (math.pi /180)*lat  #=> 2D matrix
    #1-2) calculate the inverse relative distance Earth-Sun, d_r (FAO eq#23)
    d_r = 1+0.033*np.cos(2*math.pi*doy/365)  #=> constant . fn(day)
    #1-3) calculate solar declimation, s_d (FAO eq#24)
    s_d = 0.409*np.sin(2*math.pi*doy/365-1.39) #=> constant . fn(day)
    #1-4) calculate sunset hour angle (FAO eq#25)
    sha = np.arccos(-1 * np.multiply(np.tan(lat_rad), np.tan(s_d)))  #=>2D matrix . fn(lat, day)
    #1-5) calculate Ra(extraterrestrial radiation) (FAO eq#21) unit: MJ/m2/day
    #solar constant = 0.0820 MJm^(-2) min^(-1)
    temp = np.multiply(sha, np.sin(lat_rad))
    temp = np.multiply(temp, np.sin(s_d))
    temp2 = np.multiply(np.cos(lat_rad), np.cos(s_d))
    temp2 = np.multiply(temp2, np.sin(sha))
    temp3 = np.add(temp, temp2)
    Ra = np.multiply(24*60*0.0820*d_r/math.pi, temp3)  #unit MJ/m2/day

    #2) compute Reference ET based on Hargreaves equation (FAO eq #52)
    ah=0.0023  # the Hargreaves coefficient.
    bh=0.408 # the value of 0.408 is the inverse of the latent heat flux of vaporization at 20 ◦C, 
            #changing the extraterrestrial radiation units from MJ m−2 day−1 into mm day−1 of evaporation equivalent
    TD_ch = np.subtract(Tmaximum, Tminimum)
    TD_ch = np.sqrt(TD_ch)
    temp = np.add(Tavg, 17.8)
    temp = np.multiply(ah*bh, temp)
    ETref = np.multiply(temp, TD_ch)
    ETref = np.multiply(ETref, Ra)
    return ETref

def read_AGMERRA_txt(fname, LL_lat, UR_lat, LL_lon, UR_lon, dx1, dx2):
    w_data = np.genfromtxt(fname)#, dtype="i5")#, delimiter="  ")

    w_data[w_data > 10000] = np.nan
    w_data = np.true_divide(w_data, 100) #return to original scale
    # plt.title('CHIRPS81250-original 0.05 deg')
    # plt.imshow(w_data) #, origin='upper', cmap='jet')
    # plt.colorbar()
    # plt.show()

    #extend the original coverage of clipped agmerra (13.125,17.875) and (-92.875, -88.125) to outside of the original target box "LL_lat, UR_lat, LL_lon, UR_lon"
    w_data = np.concatenate((w_data[:,0].reshape((w_data.shape[0],1)), w_data), axis=1)
    w_data = np.concatenate((w_data, w_data[:,-1].reshape((w_data.shape[0],1))), axis=1)
    w_data = np.concatenate((w_data, w_data[-1,:].reshape((1,w_data.shape[1]))), axis=0)
    w_data = np.concatenate((w_data[0,:].reshape((1,w_data.shape[1])), w_data), axis=0)

    #find index for Guatemala sub-region
    LL_lat1 = LL_lat - dx1*0.5  #take slightly wider box for interpolation
    UR_lat1 = UR_lat + dx1*0.5
    LL_lon1 = LL_lon - dx1*0.5
    UR_lon1 = UR_lon + dx1*0.5

    #regrid original 0.25 deg to 1/120 deg (~1km)
    lat1 = np.arange(LL_lat1, UR_lat1+dx1, dx1)  #numpy.arange([start, ]stop, [step, ]dtype=None)
    lon1 = np.arange(LL_lon1, UR_lon1+dx1, dx1)

    lat2 = np.arange(LL_lat+dx2*0.5, UR_lat, dx2)  #numpy.arange([start, ]stop, [step, ]dtype=None)
    lon2 = np.arange(LL_lon+dx2*0.5, UR_lon, dx2)

    #nearest neighbor interpolation
    NearestInterp_fn= RegularGridInterpolator((lon1, lat1), w_data, method='nearest')

    #generate a matrix with all the combinations of lat & lon.
    points = np.meshgrid(lon2, lat2)
    flat = np.array([m.flatten() for m in points])
    out_array = NearestInterp_fn(flat.T)
    result = out_array.reshape(*points[0].shape)

    # plt.title('AGMERRA-regridded')
    # plt.imshow(result.T) #, origin='upper', cmap='jet')
    # plt.colorbar()
    # plt.show()
    return result.T

def read_CHIRPS_txt(fname, LL_lat, UR_lat, LL_lon, UR_lon, dx1, dx2):
    rain_data = np.genfromtxt(fname)#, dtype="i5")#, delimiter="  ")

    rain_data[rain_data < -9] = np.nan
    rain_data = np.true_divide(rain_data, 1000) #return to original scale
    # plt.title('CHIRPS81250-original 0.05 deg')
    # plt.imshow(rain_data) #, origin='upper', cmap='jet')
    # plt.colorbar()
    # plt.show()

    #extend the original coverage of clipped chirps (13.025,17.975) and (-92.975, -88.025) to outside of the original target box "LL_lat, UR_lat, LL_lon, UR_lon"
    rain_data = np.concatenate((rain_data[:,0].reshape((rain_data.shape[0],1)), rain_data), axis=1)
    rain_data = np.concatenate((rain_data, rain_data[:,-1].reshape((rain_data.shape[0],1))), axis=1)
    rain_data = np.concatenate((rain_data, rain_data[-1,:].reshape((1,rain_data.shape[1]))), axis=0)
    rain_data = np.concatenate((rain_data[0,:].reshape((1,rain_data.shape[1])), rain_data), axis=0)

    #find index for Guatemala sub-region
    LL_lat1 = LL_lat - dx1*0.5  #take slightly wider box for interpolation
    UR_lat1 = UR_lat + dx1*0.5
    LL_lon1 = LL_lon - dx1*0.5
    UR_lon1 = UR_lon + dx1*0.5

    #regrid original 0.25 deg to 1/120 deg (~1km)
    lat1 = np.arange(LL_lat1, UR_lat1+dx1, dx1)  #numpy.arange([start, ]stop, [step, ]dtype=None)
    # lon1 = np.arange(LL_lon1, UR_lon1+dx1, dx1)
    lon1 = np.arange(LL_lon1, UR_lon1, dx1)

    lat2 = np.arange(LL_lat+dx2*0.5, UR_lat, dx2)  #numpy.arange([start, ]stop, [step, ]dtype=None)
    lon2 = np.arange(LL_lon+dx2*0.5, UR_lon, dx2)

    #nearest neighbor interpolation
    NearestInterp_fn= RegularGridInterpolator((lon1, lat1), rain_data, method='nearest')

    #generate a matrix with all the combinations of lat & lon.
    points = np.meshgrid(lon2, lat2)
    flat = np.array([m.flatten() for m in points])
    out_array = NearestInterp_fn(flat.T)
    result = out_array.reshape(*points[0].shape)

    # plt.title('CHIRPS81250-regridded')
    # plt.imshow(result.T) #, origin='upper', cmap='jet')
    # plt.colorbar()
    # plt.show()
    return result.T
def regrid_wspell(ini_data, LL_lat, UR_lat, LL_lon, UR_lon, dx1, dx2):
    # plt.title('CHIRPS81250-original 0.05 deg')
    # plt.imshow(ini_data) #, origin='upper', cmap='jet')
    # plt.colorbar()
    # plt.show()

    #extend the original coverage of clipped chirps (13.025,17.975) and (-92.975, -88.025) to outside of the original target box "LL_lat, UR_lat, LL_lon, UR_lon"
    ini_data = np.concatenate((ini_data[:,0].reshape((ini_data.shape[0],1)), ini_data), axis=1)
    ini_data = np.concatenate((ini_data, ini_data[:,-1].reshape((ini_data.shape[0],1))), axis=1)
    ini_data = np.concatenate((ini_data, ini_data[-1,:].reshape((1,ini_data.shape[1]))), axis=0)
    ini_data = np.concatenate((ini_data[0,:].reshape((1,ini_data.shape[1])), ini_data), axis=0)

    #find index for Guatemala sub-region
    LL_lat1 = LL_lat - dx1*0.5  #take slightly wider box for interpolation
    UR_lat1 = UR_lat + dx1*0.5
    LL_lon1 = LL_lon - dx1*0.5
    UR_lon1 = UR_lon + dx1*0.5

    #regrid original 0.25 deg to 1/120 deg (~1km)
    lat1 = np.arange(LL_lat1, UR_lat1+dx1, dx1)  #numpy.arange([start, ]stop, [step, ]dtype=None)
    # lon1 = np.arange(LL_lon1, UR_lon1+dx1, dx1)
    lon1 = np.arange(LL_lon1, UR_lon1, dx1)

    lat2 = np.arange(LL_lat+dx2*0.5, UR_lat, dx2)  #numpy.arange([start, ]stop, [step, ]dtype=None)
    lon2 = np.arange(LL_lon+dx2*0.5, UR_lon, dx2)

    #nearest neighbor interpolation
    NearestInterp_fn= RegularGridInterpolator((lon1, lat1), ini_data, method='nearest')

    #generate a matrix with all the combinations of lat & lon.
    points = np.meshgrid(lon2, lat2)
    flat = np.array([m.flatten() for m in points])
    out_array = NearestInterp_fn(flat.T)
    result = out_array.reshape(*points[0].shape)
    return result.T    
#==============================================================================================
#========Main body to call SWB function =======================================================
start_time = time.perf_counter() # => not compatible with Python 2.7

# Wdir = 'C:\\Users\Eunjin\\IRI\\CWP_Guatemala\\SWB_modeling\\test_sub_south'
# Wdir = 'C:\\Users\Eunjin\\IRI\\CWP_Guatemala\\SWB_modeling\\test_onset_1981'
Wdir = 'C:\\Users\Eunjin\\IRI\\CWP_Guatemala\\SWB_modeling\\weekly_output_2009'
os.chdir(Wdir)

#input data read from GIS shape files (gridded)
#<<====soil input from SoilGrids1km => standard 6 layers
# sdepth1 = 50.0 #[mm] thickness of the first layer 
# sdepth2 = 100.0 #[mm]
# sdepth3 = 150.0 #[mm]
# sdepth4 = 300.0 #[mm]
# sdepth5 = 400.0 #[mm]
# sdepth6 = 1000.0 #[mm]thickness of the 6th layer (200-100)

#===============READ 1km soil properties => this will be base frame for gridded swb simulation
##fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SLLL_1.asc'
##data = np.genfromtxt(fname,skip_header=6)#, dtype="i5")#, delimiter="  ")
#EJ(4/11/2019)
#read C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\TAWC_60cm.tif, instead of reaing SLLL and SDUL for each separate layer
fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\TAWC_60cm.asc'
TAWC = np.genfromtxt(fname,skip_header=6)#, dtype="i5")#, delimiter="  ")
TAWC[TAWC < 0] = np.nan
# plt.title('TAWC-60cm (1km)')
# plt.imshow(TAWC) #, origin='upper', cmap='jet')
# plt.colorbar()
# plt.show()

# #read SLLL and SDUL, and then compute SLLL_mm and SDUL_mm for the effective rooting depth (consistent to TAWC above)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SLLL_1.asc'
# SLLL1 = np.genfromtxt(fname,skip_header=6)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SLLL_2.asc'
# SLLL2 = np.genfromtxt(fname,skip_header=6)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SLLL_3.asc'
# SLLL3 = np.genfromtxt(fname,skip_header=6)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SLLL_4.asc'
# SLLL4 = np.genfromtxt(fname,skip_header=6)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SDUL_1.asc'
# SDUL1 = np.genfromtxt(fname,skip_header=6)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SDUL_2.asc'
# SDUL2 = np.genfromtxt(fname,skip_header=6)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SDUL_3.asc'
# SDUL3 = np.genfromtxt(fname,skip_header=6)
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\ISRIC_soil_1km\SDUL_SLLL_clipped_GT\SDUL_4.asc'
# SDUL4 = np.genfromtxt(fname,skip_header=6)
# SDUL_mm = SDUL1*sdepth1 + SDUL2*sdepth2 + SDUL3*sdepth3 + SDUL4*sdepth4 #+ SDUL5*sdepth5  #convert SDUL [cm3/cm3] to mm for whole soil profile 
# SLLL_mm = SLLL1*sdepth1 + SLLL2*sdepth2 + SLLL3*sdepth3 + SLLL4*sdepth4 #+ SLLL5*sdepth5  #convert SLLL [cm3/cm3] to mm for whole soil profile 

ncols=600  #<==TAWC_60cm.asc'
nrows=600   #<==TAWC_60cm.asc'
# xllcorner = -93.000000000000  #<==TAWC_60cm.asc'     #-93.000000000028  
# yllcorner = 13.000000000000  #<==TAWC_60cm.asc'   12.999999999988
# cellsize = 0.008333333333

#13) ===========ADD DEM (1km)
fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\GTOPO30_DEM\gt30w100n40_clip.asc'
DEM_data = np.genfromtxt(fname,skip_header=6)
# DEM_data[DEM_data < 0] = np.nan
# plt.title('DEM(1km)')
# plt.imshow(DEM_data) #, origin='upper', cmap='jet')
# plt.colorbar()
# plt.show()

# #14) ===========ADD UY country mask
# #fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\SWB_modeling\GT_admin1_1k_mask.asc'
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\SWB_modeling\GT_admin1_1k_mask2.asc'  #EJ(4/11/2019) updated to cover entire dry corridor
# GT_mask = np.genfromtxt(fname,skip_header=6)#, dtype="i5")#, delimiter="  ")

# #15) ===========ADD IIASA crop cover
# fname=r'C:\Users\Eunjin\IRI\CWP_Guatemala\SWB_modeling\cropland_hybrid_10042015v9\iiasa_crop_grid.asc'
# ccover_mask = np.genfromtxt(fname,skip_header=6)#, dtype="i5")#, delimiter="  ")


#==================================================
#+++++++++++++++++++++++++++++++++++++++++++++++++
#==================================================
#==============================================================================================
#===========L
#daily loop
#===========L
#input data (should be automatic for gridded modeling)
# IC_date= 2009091   # May 10 => The first sowing season starts with the first rains in late April and
#                            #extends until mid-May.    273
# start_year=2009  #the worst drought in Latin America, El Nino 2009-2010
# days_sim= 210  #April - Oct => length of simulation days
# plt_date = 2009130 #planting date

start_year=2009  #the worst drought in Latin America, El Nino 2009-2010
IC_date= start_year*1000+91
days_sim= 210  #April - Oct => length of simulation days
plt_date = start_year*1000+130
# There are two seasons for sowing maize in this region, both of which are based on the
# annual rainfall pattern. The first sowing season starts with the first rains in late April and
# extends until mid-May. The second sowing season lasts the entire month of August. The
# second rainfall season is considered riskier in terms of water availability during grain
# filling. Consequently, the great majority of farmers only plant during the first sowing
# season; nearly 90% of all the sample farmers planted their maize crop during the first two
# weeks of May. 

rho = 0.5 #fraction of plant's water holding capacity (or total available water = SDUL - SLLL)
          #readily available water =  rho * TAW
init_sm_frac= 0.3 #intitial soil moisture fraction of TAW
crop= 'maize'
#Readily Available Water (RAW_op)
#RAW = fraction of TAW that a crop can extract from the root zone without surffering water stress [FAO56, p162]
# rho normally varies from 0.3 for shallow rooted plants at high rates of ETc (> 8mm/day)
#    to 0.7 for deep rooted plants at low rates of ETc (< 3mm/d). 0.5 is commonly used for many crops
RAW = rho*TAWC  #rho = 0~1 ===> rho_adj is used varying RAW with different ETcrop (line #437)
#!Initial CONDITION at start of simulation
SMinit = init_sm_frac * TAWC  #[mm]
Loss= np.zeros((TAWC.shape[0],TAWC.shape[1])) #deep drainage, percolation

#!end variable assigments
#!crop parameter assignment
end_time0 = time.perf_counter() # => not compatible with Python 2.7
print('It took {0:5.1f} sec to read static input (before daily loop)'.format(end_time0-start_time))

#*********************************************************************************
#compute daily Kc values for each pixel based on spatially varying onset dates
#1) compute planting date (plt date = onset date + initial wet spell
##read dat
#read a CHIRPS sample data for mask
fname='C:\\Users\\Eunjin\\IRI\\CWP_Guatemala\\CHIRPS_GT\\CHIRPS_GT_extracted\\CHIRPS15334.txt'
mask_data = np.genfromtxt(fname)#, dtype="i5")#, delimiter="  ")
# fname= r'C:\Users\Eunjin\IRI\CWP_Guatemala\Onset\length_wet_spell.npy'
fname = 'C:\\Users\\Eunjin\\IRI\\CWP_Guatemala\\Onset\\length_wet_spell_MJJASO.npy'
data=np.load(fname)
#1) long-term average
avg_w_spell = np.nanmean(data, axis=2)
avg_w_spell[mask_data < 0] = np.nan
#2) long-term stdev
std_w_spell = np.nanstd(data, axis=2)
std_w_spell[mask_data < 0] = np.nan
#3) length of initial wet spell
ini_wspell = np.add(avg_w_spell, std_w_spell)
ini_wspell = np.ceil(ini_wspell)  #**<=======================
LL_lat = 13
UR_lat = 18
LL_lon = -93
UR_lon = -88
dx_ch = 0.05   #deg  approximately 5km (CHIRPS resolution)
dx = 1/120  #approximately 1km
ini_wspell_1km = regrid_wspell(ini_wspell, LL_lat, UR_lat, LL_lon, UR_lon, dx_ch, dx)  #regrid from 0.05 deg to 1/120 (1km) resolution

##read onset date
# fname = 'C:\\Users\\Eunjin\\IRI\\CWP_Guatemala\\Onset\\Estimated_Onset\\onset_' + str(start_year) +'.npy'
fname = 'C:\\Users\\Eunjin\\IRI\\CWP_Guatemala\\Onset\\Estimated_Onset_7d\\onset_' + str(start_year) +'.npy'
onset =np.load(fname)

plt_date = np.add(onset, ini_wspell_1km)  #*** <<++++++++++++++++++++++++++++++++++++
# plt.title('Planting date (onset + initial wet spell) - %d' %(start_year))
# plt.imshow(data) #, origin='upper', cmap='jet')
# plt.colorbar()
# plt.show()
#Q: How to handle nan values?

#====================================================================================
#!crop parameter assignment (need to change later so as to get these from input file)
kinit=0.3
kmid=1.2
kend =0.6
init_len=30
veg_len=30
mid_len=30
late_len=30
#make empty 3D arrays to save Kc
Kc_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float)

for k in range(nrows):
    for j in range(ncols):
        day1=IC_date%1000  #91
        day2 = plt_date[k,j] -4 #assuming plowing 4 days before planting, when rain starts
        day11 = day2 -1
        day3 = plt_date[k,j]  #planting date = onset + initial wet spell
        day4 = plt_date[k,j] + init_len
        day5 = plt_date[k,j] + init_len + veg_len
        day6 = plt_date[k,j] + init_len + veg_len + mid_len
        day7 = plt_date[k,j] + init_len + veg_len + mid_len + late_len
        day8 = day7 + 1
        day9 = day1 + days_sim - 1  #330
        xp = [day1, day11, day2, day3, day4, day5, day6, day7, day8, day9]
        fp = [1, 1, kinit, kinit,kinit, kmid, kmid,kend, 1, 1]
        x_interp = np.arange(day1+1,day9) # 92~ 329 => 238 days
        
        Kc_interp = np.interp(x_interp, xp, fp)  #shape = 238
        temp = np.reshape(Kc_interp, (1,Kc_interp.shape[0]))
        temp2 = np.concatenate(([[1.0]], temp), axis=1)   #add the first Kc element (default 1.0) in the beginning of the Kc array
        temp3 = np.concatenate((temp2,[[1.0]]), axis=1)   #add the last Kc element (default 1.0) to the Kc array
        Kc_3D[k,j,:] = temp3

end_time0 = time.perf_counter() # => not compatible with Python 2.7
print('It took {0:5.1f} sec to compute Kc'.format(end_time0-start_time))
# plt.title('Kc Values on DOY=180, 1981')
# plt.imshow(Kc_3D[:,:,90]) #, origin='upper', cmap='jet')
# plt.colorbar()
# plt.clim(0,1)
# plt.show()

#!========start a loop (subprocess)=================================================
day_count = 1
# planted = 'false' #initialize
##idoy = plt_doy #for current doy within the loop
cur_doy = IC_date%1000

#make empty 3D arrays to save outputs
pTAW = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float)  ##percentage of current soil water content out of TAW
pRAW = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float)  #!above or below theta_RAW, threshold where water stress starts
ETref = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
Peff_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
SMnow_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
ETcrop_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
ETc_act_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
Drainage_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
Runoff_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
rho_adj_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
Ks_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
RAW_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float) 
WRSI_3D = np.empty((TAWC.shape[0],TAWC.shape[1], days_sim), dtype=float)   #Water Requirement Satisfaction Index

#read onset date
##read dat
fname = 'C:\\Users\\Eunjin\\IRI\\CWP_Guatemala\\Onset\\Estimated_Onset\\onset_' + str(start_year) +'.npy'
onset = np.load(fname)

while day_count <= days_sim:  #MAYBE DURING croplen (CROP GROWING PERIOD)
   #===========LOAD rainfall from CHIRPIS: Note: this data is already clipped to Guatemala
   #paths to the saved daily weather data
   CHIRPS_dir='C:\\Users\\Eunjin\\IRI\\CWP_Guatemala\\CHIRPS_GT\\CHIRPS_GT_extracted\\'
   fname = CHIRPS_dir + 'CHIRPS' + str(start_year)[2:] + repr(cur_doy).zfill(3) + '.txt'  #CHIRPS81001.txt
   dx1 = 0.05   #deg  approximately 5km
   dx2 = 1/120 #approximately 1km
   rain_day = read_CHIRPS_txt(fname, LL_lat, UR_lat, LL_lon, UR_lon, dx1, dx2)
   # plt.title('CHIRPS81250')
   # plt.imshow(rain_day) #, origin='upper', cmap='jet')
   # plt.colorbar()
   # plt.show()

   #===========LOAD tmin,tmax, srad from AGMERRA: Note: this data is already clipped to Guatemala
   AGMERRA_dir='C:\\Users\\Eunjin\\IRI\\CWP_Guatemala\\AGMERRA_GT_025\\AGMERRA_Temp_Srad\\'
   dx1 = 0.25   #deg  approximately 25km
   # dx2 = 1/120 #approximately 1km
   fname = AGMERRA_dir + 'Srad\\agmerra_SRad_' + str(start_year)[2:] + repr(cur_doy).zfill(3) + '.txt'  #agmerra_SRad_00001.txt
   srad_day = read_AGMERRA_txt(fname, LL_lat, UR_lat, LL_lon, UR_lon, dx1, dx2)

   fname = AGMERRA_dir + 'Tmin\\agmerra_Tmin_' + str(start_year)[2:] + repr(cur_doy).zfill(3) + '.txt'  #agmerra_Tmin_00001.txt
   tmin_day = read_AGMERRA_txt(fname, LL_lat, UR_lat, LL_lon, UR_lon, dx1, dx2)

   fname = AGMERRA_dir + 'Tmax\\agmerra_Tmax_' + str(start_year)[2:] + repr(cur_doy).zfill(3) + '.txt'  #agmerra_Tmax_00001.txt
   tmax_day = read_AGMERRA_txt(fname, LL_lat, UR_lat, LL_lon, UR_lon, dx1, dx2)

   # plt.title('AGMERRA-Tmax regridded')
   # plt.imshow(tmax_day) #, origin='upper', cmap='jet')
   # plt.colorbar()
   # plt.show()

   # end_time0 = time.perf_counter() # => not compatible with Python 2.7
   # print('It took {0:5.1f} sec to read tmin'.format(end_time0-start_time))


   #1) CALCULATE EFFECTIVE PRECIPITATION
   #typical CN range for Agricultural => https://engineering.purdue.edu/mapserve/LTHIA7/documentation/scs.htm
   #Hydrologic group A(64), B(75), C(82), D(85) =>Group A is sand, loamy sand or sandy loam types of soils. It has low runoff potential and high infiltration rate
   # Group B is silt loam or loam.
   CN = 75
   Peff, Runoff = Peffective_2D(rain_day, CN)
   # plt.title('effective precipitation')
   # plt.imshow(Peff) #, origin='upper', cmap='jet')
   # plt.colorbar()
   # plt.show()

   #2) CALCULATE ETref (for reference plant, grass)
   lat2 = np.arange(LL_lat+dx2*0.5, UR_lat, dx2)  #numpy.arange([start, ]stop, [step, ]dtype=None)
   lon2 = np.arange(LL_lon+dx2*0.5, UR_lon, dx2)
   # ETref = ETref_PM_2D(lat2,cur_doy,DEM_data,srad_day,tmax_day,tmin_day)  #FAO Penman-Monteith 
   # #
   ETref_H = ETref_Harg_2D(lat2,cur_doy,DEM_data,tmax_day,tmin_day)  #FAO Hargreaves method

   # end_time0 = time.perf_counter() # => not compatible with Python 2.7
   # print('It took {0:5.1f} sec to compute ET_Hargreaves'.format(end_time0-start_time))

   # plt.title('Refernce ET - Hargreaves')
   # plt.imshow(ETref_H) #, origin='upper', cmap='jet')
   # plt.colorbar()
 
   #ETcrop (actual ET) ==> potential ETcrop without any water stress
#    ETcrop=Kc*ETref_H     #ETcrop = mm/day 
   ETcrop = np.multiply(Kc_3D[:,:,day_count-1], ETref_H)
   
   #4)Adjust rho (depletion factor, RAW = rho * TAW)
   # => rho can be a constant (e.g., 0.5), but can be a function of ETc
   #Vary rho for RAW depending on weather (dry or wet) and ETcrop
   #numerical approximation for adjusting rho for ETcrop rate [FAO p.162]
   rho_maize=0.55 #<<<<<<<=======from Table 22 on page 163 ==>shoud be changed automatically later
   rho_adj=rho_maize + 0.04*(5-ETcrop)  #
   rho_adj[rho_adj < 0.1]=0.1
   rho_adj[rho_adj > 0.8]=0.8
   # if rho_adj < 0.1:  #adjusted rho should be limted to 0.1~0.8 and ETcrop is in mm/day
   #     rho_adj=0.1
   # elif rho_adj > 0.8:
   #     rho_adj=0.8
   
   # RAW=rho_adj*TAWC  #rho = 0~1
   RAW = np.multiply(rho_adj, TAWC)

   #Adjust ETcrop under soil water stress condition (using single coefficient approach of FAO56)
   #Refer figure 42 from FAO56 page 167
   #NOTE: SMinit is amount of water relative to TAW. Therefore for absolute water content (like SDUL), need to add SLLL
   Ks = np.zeros((RAW.shape[0],RAW.shape[1])) + 1.0
   temp = np.subtract(TAWC, RAW)
   slope = np.divide(1,temp) 
   temp3 = np.multiply(slope, SMinit)
   Ks = np.where(SMinit < temp, temp3, Ks)

   # temp = np.add(SMinit,SLLL_mm)
   # temp2 = np.subtract(SDUL_mm, RAW)
   # # slope = np.divide(1,(SDUL_mm - RAW - SLLL_mm))  #1/(TAWC-RAWC)
   # slope = np.divide(1,(TAWC - RAW)) 
   # # temp3 = np.multiply(slope, SMinit)
   # temp3 = np.multiply(slope, (SMinit+SLLL_mm))
   # Ks = np.where(temp < temp2, temp3, Ks)
   # if (SMinit + SLLL_mm) > SDUL_mm:
   #     Ks=1.0
   # elif (SMinit + SLLL_mm) <= SDUL_mm and (SMinit + SLLL_mm) >= (SDUL_mm - RAW):
   #     Ks=1.0
   # else:  #linearly decrease  => Refer figure 42 from FAO56 page 167
   #     slope=1/(SDUL_mm - RAW - SLLL_mm)  #theta_t = SDUL_mm-RAW
   #     Ks=slope * SMinit
   # #precaution
   # if Ks < 0 or Ks > 1:
   #     tkMessageBox.showerror('Error in Ks', 'Estimated Ks is beyond the range!')
   #     os.system("pause")

   #!Reduced crop ET
   # ETcrop_act=Ks*ETcrop
   ETcrop_act = np.multiply(Ks, ETcrop)


   #!Simplified Water Balance Equation
   SMnew = SMinit + Peff - ETcrop_act
   SMnow = np.copy(SMnew)
   SMnow = np.where(SMnew > TAWC, TAWC, SMnew)
   Loss = np.subtract(SMnew, TAWC)  #drainage, percolation loss
   SMnow[SMnow < 0] = 0.0
   SMnow = np.where(SMnow > TAWC, TAWC, SMnow)
   # if SMnew > TAW:
   #     SMnow = TAW
   #     Loss = SMnew - TAW  #this is deep vertical drainage (1-layer soil water balance)
   # else:
   #     SMnow=SMnew
   #     if SMnow < 0.0:
   #         SMnow = 0.0
   # end_time0 = time.perf_counter() # => not compatible with Python 2.7
   # print('It took {0:5.1f} sec to compute updated soil moisture'.format(end_time0-start_time))

   temp = np.divide(SMnow,TAWC)
   temp[np.isnan(TAWC)]= np.nan  #mask with available TAWC pixels
   pTAW[:,:,day_count-1] = temp ##percentage of current soil water content out of TAW
   
   temp = SMnow-(TAWC-RAW) #!above or below theta_RAW, threshold where water stress starts
   temp[np.isnan(TAWC)]= np.nan  #mask with available TAWC pixels
   pRAW[:,:,day_count-1] = temp 
   ETref[:,:,day_count-1] = ETref_H  
   Peff_3D[:,:,day_count-1] = Peff 
   SMnow[np.isnan(TAWC)]=np.nan
   SMnow_3D[:,:,day_count-1] = SMnow 
   ETcrop[np.isnan(TAWC)]= np.nan 
   ETcrop_3D[:,:,day_count-1] = ETcrop 
   ETcrop_act[np.isnan(TAWC)]= np.nan 
   ETc_act_3D[:,:,day_count-1] = ETcrop_act 
   #WRSI
   if day_count == 1:
      WRSI = np.divide(ETcrop_act, ETcrop)
      WRSI[np.isnan(ETcrop_act)]=np.nan
      WRSI[np.isnan(TAWC)]=np.nan #mask with available TAWC pixels
      WRSI_3D[:,:,day_count-1] = WRSI 
   else:
      temp1 = np.nansum(ETcrop_3D,axis = 2)
      temp2 = np.nansum(ETc_act_3D,axis = 2)
      WRSI = np.divide(temp2, temp1)  #current WRSI
      WRSI[np.isnan(ETcrop_act)]=np.nan
      WRSI[np.isnan(TAWC)]=np.nan #mask with available TAWC pixels
      WRSI_3D[:,:,day_count-1] = WRSI 

   Loss[np.isnan(TAWC)]=np.nan  #mask with available TAWC pixels
   Drainage_3D[:,:,day_count-1] = Loss  
   Runoff[np.isnan(TAWC)]=np.nan #mask with available TAWC pixels
   Runoff_3D[:,:,day_count-1] = Runoff 
   rho_adj[np.isnan(TAWC)]=np.nan #mask with available TAWC pixels
   rho_adj_3D[:,:,day_count-1] = rho_adj 
   Ks[Ks < 0]=np.nan
   Ks[np.isnan(TAWC)]=np.nan 
   Ks_3D[:,:,day_count-1] = Ks 
   RAW[np.isnan(TAWC)]= np.nan 
   RAW_3D[:,:,day_count-1] = RAW 
   #!initialize SM for next timestep
   SMinit = np.copy(SMnow)
   # Loss = 0.0  #!remember this!!!  ???
   Loss= np.zeros((TAWC.shape[0],TAWC.shape[1]))

   day_count = day_count + 1
   #update current DOY (and start_year if needed)
   if calendar.isleap(start_year):
      if cur_doy == 366:
         start_year = start_year + 1
         cur_doy = 1
      else:
         cur_doy = cur_doy + 1
   else:
      if cur_doy == 365:
         start_year = start_year + 1
         cur_doy = 1
      else:
         cur_doy = cur_doy + 1

   # end_time0 = time.perf_counter() # => not compatible with Python 2.7
   # print('It took {0:5.1f} sec to save daily ouput'.format(end_time0-start_time))
   # print('end of day {0:5.1f} computation'.format(day_count))

#=========================================================================
#=========================================================================
#make weekly aggregation
week_list = []
month_list =[]
day_list = []
doy_list = []
count = 0
count = 0
for idoy in range(IC_date%1000, IC_date%1000 + days_sim):
   a = datetime.strptime('{0}'.format(start_year), '%Y') + timedelta(days=idoy)
   # imonth = int(a.strftime('%m'))
   # iday = int(a.strftime('%d'))
   # iweek = int(a.strftime("%U"))
   month_list.append(int(a.strftime('%m')))
   day_list.append(int(a.strftime('%d')))
   week_list.append(int(a.strftime("%U")))
   doy_list.append(idoy)
   count +=count

#write into csv file
with open('week_day_list.csv', mode='w', newline='') as outfile:
    csv_writer = csv.writer(outfile) #, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(['Month', 'Day', 'DOY','No_Week'])
    for i in range(len(week_list)):
       csv_writer.writerow([month_list[i], day_list[i], doy_list[i],week_list[i]])

week_array = np.asarray(week_list)
w_unique = np.unique(week_array)

#Make empty 3D arrays
pTAW_week = np.empty((TAWC.shape[0],TAWC.shape[1], w_unique.shape[0]), dtype=float) 
pRAW_week = np.empty((TAWC.shape[0],TAWC.shape[1], w_unique.shape[0]), dtype=float)
ETref_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
Peff_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
SMnow_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
ETcrop_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
ETc_act_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
Drainage_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
Runoff_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
rho_adj_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
Ks_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
RAW_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
WRSI_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
WRSI_week2 = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)
Kc_week = np.empty((TAWC.shape[0],TAWC.shape[1],w_unique.shape[0]), dtype=float)

#weekly avg
w_counter = 0
for i in w_unique:
   temp = np.where(week_array == i)
   index1 = temp[0][0]
   index2 = temp[0][-1]
   pTAW_week[:,:,w_counter] = np.nanmean(pTAW[:,:,index1:index2+1], axis=2)
   pRAW_week[:,:,w_counter] = np.nanmean(pRAW[:,:,index1:index2+1], axis=2)
   ETref_week[:,:,w_counter] = np.nanmean(ETref[:,:,index1:index2+1], axis=2)
   Peff_week[:,:,w_counter] = np.nanmean(Peff_3D[:,:,index1:index2+1], axis=2)
   SMnow_week[:,:,w_counter] = np.nanmean(SMnow_3D[:,:,index1:index2+1], axis=2)
   ETcrop_week[:,:,w_counter] = np.nanmean(ETcrop_3D[:,:,index1:index2+1], axis=2)
   ETc_act_week[:,:,w_counter] = np.nanmean(ETc_act_3D[:,:,index1:index2+1], axis=2)
   Drainage_week[:,:,w_counter] = np.nanmean(Drainage_3D[:,:,index1:index2+1], axis=2)
   Runoff_week[:,:,w_counter] = np.nanmean(Runoff_3D[:,:,index1:index2+1], axis=2)
   rho_adj_week[:,:,w_counter] = np.nanmean(rho_adj_3D[:,:,index1:index2+1], axis=2)
   Ks_week[:,:,w_counter] = np.nanmean(Ks_3D[:,:,index1:index2+1], axis=2)
   RAW_week[:,:,w_counter] = np.nanmean(RAW_3D[:,:,index1:index2+1], axis=2)
   WRSI_week[:,:,w_counter] = np.nanmean(WRSI_3D[:,:,index1:index2+1], axis=2)  #cumulative from the simulation start
   Kc_week[:,:,w_counter] = np.nanmean(Kc_3D[:,:,index1:index2+1], axis=2)
   WRSI_week2[:,:,w_counter] = np.divide(ETc_act_week[:,:,w_counter], ETcrop_week[:,:,w_counter])  #current WEEk WRSI
   w_counter = w_counter +1

   # plt.imshow(pTAW_week[:,:,w_counter-1]) #, origin='upper', cmap='jet')
   # plt.colorbar()
   # # plt.clim(0,1)
   # plt.show()

#save 3D output
# np.save('pTAW_week.npy', pTAW_week)
# np.save('pRAW_week.npy', pRAW_week)
# np.save('ETref_week.npy', ETref_week)
# np.save('Peff_week.npy', Peff_week)
# np.save('SMnow_week.npy', SMnow_week)
# np.save('ETcrop_week.npy', ETcrop_week)
# np.save('ETact_week.npy', ETc_act_week)
# # np.save('Drainage_week.npy', Drainage_week)
# # np.save('Runoff_week.npy', Runoff_week)
# # np.save('rho_adj_week.npy', rho_adj_week)
# # np.save('Ks.npy_week', Ks_week)
# # np.save('RAW_week.npy', RAW_week)
# np.save('WRSI_week.npy', WRSI_week)
# np.save('Kc_week.npy', Kc_week)
np.save('WRSI_week2.npy', WRSI_week2)

end_time0 = time.perf_counter() # => not compatible with Python 2.7
print('It took {0:5.1f} sec to save all final weekly outputs'.format(end_time0-start_time))
print('********End of seasonal simulation!!!')
