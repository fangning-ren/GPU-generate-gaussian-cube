import re
import sys

class Excitation:
    def __init__(self):
        self.orb1 = [1,]
        self.orb2 = [2,]
        self.cisc = [1.,]
        self.osci = 1.0
        self.e = 1.0
        self.wlen = 45.5640
        self.Tx, self.Ty, self.Tz, self.T2 = 0.0, 0.0, 0.0, 0.0
        self.vTx, self.vTy, self.vTz, self.vT2 = 0.0, 0.0, 0.0, 0.0


def read_cis_output(fp:str):
    with open(fp, "r", errors="ignore") as f:
        for i in range(100000):
            line = f.readline()
            if line.find("Transition dipole moments") != -1:
                curlinestart = f.tell()
                break
        # reading transition dipole
        f.seek(curlinestart)
        dipoledata = []
        patt = r'(?P<ridx>\d+) +(?P<Tx>-?\d+\.\d+) +(?P<Ty>-?\d+\.\d+) +(?P<Tz>-?\d+\.\d+) +(?P<T>-?\d+\.\d+)'
        for i in range(100):
            line = f.readline()
            if line.find("Transition dipole moments between excited states:") != -1:
                curlinestart = f.tell()
                break
            result = re.search(patt, line)
            if not result:
                continue
            result = result.groupdict()
            dipoledata.append([float(result["Tx"]), float(result["Ty"]), float(result["Tz"]), float(result["T"])**2])

        # skipping infos
        f.seek(curlinestart)
        for i in range(10000):
            curlinestart = f.tell()
            line = f.readline()
            if line.find("Velocity transition dipole moments") != -1:
                curlinestart = f.tell()
                break

        # reading velocity dipole
        f.seek(curlinestart)
        vpoledata = []
        patt = r'(?P<ridx>\d+) +(?P<Tx>-?\d+\.\d+) +(?P<Ty>-?\d+\.\d+) +(?P<Tz>-?\d+\.\d+) +(?P<T>-?\d+\.\d+)'
        for i in range(100):
            line = f.readline()
            if line.find("Magnetic transition dipole moments and rotational strengths") != -1:
                curlinestart = f.tell()
                break
            result = re.search(patt, line)
            if not result:
                continue
            result = result.groupdict()
            vpoledata.append([float(result["Tx"]), float(result["Ty"]), float(result["Tz"]), float(result["T"])**2])

        # skipping infos
        f.seek(curlinestart)
        for i in range(10000):
            curlinestart = f.tell()
            line = f.readline()
            if line.find("Largest CI coefficients") != -1:
                break

        # reading CI coefficient
        f.seek(curlinestart)
        occus, virts, coeffs = [], [], []
        patt = r"(?P<occu>\d+) +-> +(?P<virt>\d+) +:.+ +(?P<coeff>-?\d+\.\d+)"
        for i in range(1000):
            curlinestart = f.tell()
            line = f.readline()
            if line.find("Final Excited State Results:") != -1:
                break
            if line.find("Largest CI coefficients") != -1:
                occus.append([])
                virts.append([])
                coeffs.append([])
            result = re.search(patt, line)
            if not result:
                continue
            result = result.groupdict()
            occus[-1].append(int(result["occu"])-1)
            virts[-1].append(int(result["virt"])-1)
            coeffs[-1].append(float(result["coeff"]))
        f.seek(curlinestart)
        eexts, oscis = [], []
        for i in range(1000):
            line = f.readline()
            if not line:
                break
            patt = r"(?P<ridx>\d+) +(?P<tene>-?\d+\.\d+) +(?P<eext>-?\d+\.\d+) +(?P<osci>-?\d+\.\d+) +(?P<s2>-?\d+\.\d+)"
            result = re.search(patt, line)
            if not result:
                continue
            eexts.append(float(result["eext"]) / 27.21139664130791)
            oscis.append(float(result["osci"]))
    
    if not (len(dipoledata) == len(occus) == len(virts) == len(coeffs) == len(eexts) == len(oscis)):
        raise ValueError("Errors found in reading excitation info.")

    excitations = []
    for i in range(len(eexts)):
        e = Excitation()
        e.cisc = coeffs[i]
        e.e = eexts[i]
        if not (len(coeffs[i]) == len(occus[i]) == len(virts[i])):
            raise ValueError("Errors found in reading excitation info.")
        e.osci = oscis[i]
        e.orb1 = occus[i]
        e.orb2 = virts[i]
        e.Tx, e.Ty, e.Tz, e.T2 = dipoledata[i]
        e.vTx, e.vTy, e.vTz, e.vT2 = vpoledata[i]
        e.wlen = 45.56337117 / e.e
        excitations.append(e)
    return excitations

def write_multiwfn_readable_orca_output_file(fname:str, extlist):
    """Generate a orca type output file whose excitations can be readed by multiwfn"""
    occus, virts = [], []
    for ext in extlist:
        occus += list(ext.orb1)
        virts += list(ext.orb2)
    
    f = open(fname, "w")
    f.write("""
    
                                 *****************
                                 * O   R   C   A *
                                 *****************

                                            #,                                       
                                            ###                                      
                                            ####                                     
                                            #####                                    
                                            ######                                   
                                           ########,                                 
                                     ,,################,,,,,                         
                               ,,#################################,,                 
                          ,,##########################################,,             
                       ,#########################################, ''#####,          
                    ,#############################################,,   '####,        
                  ,##################################################,,,,####,       
                ,###########''''           ''''###############################       
              ,#####''   ,,,,##########,,,,          '''####'''          '####       
            ,##' ,,,,###########################,,,                        '##       
           ' ,,###''''                  '''############,,,                           
         ,,##''                                '''############,,,,        ,,,,,,###''
      ,#''                                            '''#######################'''  
     '                                                          ''''####''''         
             ,#######,   #######,   ,#######,      ##                                
            ,#'     '#,  ##    ##  ,#'     '#,    #''#        ######   ,####,        
            ##       ##  ##   ,#'  ##            #'  '#       #        #'  '#        
            ##       ##  #######   ##           ,######,      #####,   #    #        
            '#,     ,#'  ##    ##  '#,     ,#' ,#      #,         ##   #,  ,#        
             '#######'   ##     ##  '#######'  #'      '#     #####' # '####'        

    \n""")
    f.write(""" 
       ***********************************************************
       *                    TeraChem v1.9-2021.10-dev            *
       *                 Hg Version:                             *
       *                   Development Version                   *
       *           Chemistry at the Speed of Graphics!           *
       ***********************************************************
       *        The original data is generated by TeraChem       *
       *             Converted by read_excitation.py             *
       ***********************************************************

------------------------------------------------------------------------------
            THIS OUTPUT ONLY CONTAINS EXCITATION CI COEFFICIENTS
                        OTHER INFORMATION ARE INVALID
------------------------------------------------------------------------------
    \n""")

    f.write(f"""
------------------------------------------------------------------------------
                        ORCA TD-DFT/TDA CALCULATION
------------------------------------------------------------------------------

Input orbitals are from        ... {fname}.gbw
CI-vector output               ... {fname}.cis
Tamm-Dancoff approximation     ... operative
CIS-Integral strategy          ... AO-integrals
Integral handling              ... AO integral Direct
Max. core memory used          ... 2048 MB
Reference state                ... RHF
Generation of triplets         ... off
Follow IRoot                   ... off
Number of operators            ... 1
Orbital ranges used for CIS calculation:
 Operator 0:  Orbitals  0...{max(occus)}  to {min(virts)}...1631
XAS localization array:
 Operator 0:  Orbitals  -1... -1
    \n""")

    f.write("""
     *** TD-DFT CALCULATION INITIALIZED ***

------------------------
DAVIDSON-DIAGONALIZATION
------------------------

Dimension of the eigenvalue problem            ... 131072
Number of roots to be determined               ...      5
Maximum size of the expansion space            ...     50
Maximum number of iterations                   ...    100
Convergence tolerance for the residual         ...    2.500e-07
Convergence tolerance for the energies         ...    2.500e-07
Orthogonality tolerance                        ...    1.000e-14
Level Shift                                    ...    0.000e+00
Constructing the preconditioner                ... o.k.
Building the initial guess                     ... o.k.
Number of trial vectors determined             ...     50


                       ****Iteration    0****

   Memory handling for direct AO based CIS:
   Memory per vector needed      ...   128 MB
   Memory needed                 ...  2048 MB
   Memory available              ...  2048 MB
   Number of vectors per batch   ...    16
   Number of batches             ...     2
   Time for densities:            0.000
   Time for RI-J (Direct):        0.000
   Time for K (COSX):             0.000
   Time for XC-Integration:       0.000
   Time for Sigma-Completion:     0.000
   Time for densities:            0.000
   Time for RI-J (Direct):        0.000
   Time for K (COSX):             0.000
   Time for XC-Integration:       0.000
   Time for Sigma-Completion:     0.000
   Size of expansion space: 15
   Lowest Energy          :     0.000000000000
   Maximum Energy change  :     0.000000000000 (vector 1)
   Maximum residual norm  :     0.000000000000


      *** CONVERGENCE OF RESIDUAL NORM REACHED ***

Storing the converged CI vectors               ... 7-coronene-hh-tddft.cis1

                 *** DAVIDSON DONE ***

Total time for solving the CIS problem:   702.861sec
    \n""")
    f.write("""
------------------------------------
TD-DFT/TDA EXCITED STATES (SINGLETS)
------------------------------------

the weight of the individual excitations are printed if larger than 1.0e-02""")
    for i, ext in enumerate(extlist):
        ext:Excitation
        f.write(f"\nSTATE  {i+1}:  E=   {ext.e:.6f} au      {ext.e*27.211:>.3f} eV    {ext.e*219474.6:>6.1f} cm**-1 <S**2> =   0.000000\n")
        for orb1, orb2, coeff in zip(ext.orb1, ext.orb2, ext.cisc):
            f.write(f"   {orb1}a -> {orb2}a  :   {coeff**2:>3.6f} (c={coeff:>3.8f})\n")

    f.write("""
-----------------------------
TD-DFT/TDA-EXCITATION SPECTRA
-----------------------------

Center of mass = (  0.0000,  0.0000,  3.4488)
Calculating the Dipole integrals                  ... done
Transforming integrals                            ... done
Calculating the Linear Momentum integrals         ... done
Transforming integrals                            ... done
Calculating angular momentum integrals            ... done
Transforming integrals                            ... done
    """)
    f.write("""
-----------------------------------------------------------------------------
         ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
-----------------------------------------------------------------------------
State   Energy    Wavelength  fosc         T2        TX        TY        TZ  
        (cm-1)      (nm)                 (au**2)    (au)      (au)      (au) 
-----------------------------------------------------------------------------
""")
    for i, ext in enumerate(extlist):
        ext:Excitation
        f.write(f"   {i+1:<2d}   {ext.e*219474.6: <6.1f}    {ext.wlen: <3.1f}   {ext.osci: <2.9f}   {ext.T2: <2.5f}  {ext.Tx: <2.5f}   {ext.Ty: <2.5f}   {ext.Tz :<2.5f}\n")

    f.write("""
-----------------------------------------------------------------------------
         ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS
-----------------------------------------------------------------------------
State   Energy    Wavelength   fosc        P2         PX        PY        PZ  
        (cm-1)      (nm)                 (au**2)     (au)      (au)      (au) 
-----------------------------------------------------------------------------
""")
    for i, ext in enumerate(extlist):
        ext:Excitation
        f.write(f"   {i+1:>2d}   {ext.e*219474.6:>6.1f}    {ext.wlen:>3.1f}   {ext.osci:>2.9f}   {ext.vT2:>2.5f}  {ext.vTx:>2.5f}   {ext.vTy:>2.5f}   {ext.vTz:>2.5f}\n")
    f.write("""
Total run time:      0.000 sec

           *** ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR ***

Maximum memory used throughout the entire CIS-calculation: 1024.0 MB

-----------------------
CIS/TD-DFT TOTAL ENERGY
-----------------------

    E(SCF)  =      0.000000000 Eh
    DE(CIS) =      0.000000000 Eh (Root  1)
    ----------------------------- ---------
    E(tot)  =      0.000000000 Eh



-------------------------   ----------------
Dispersion correction           -0.000000000
-------------------------   ----------------


-------------------------   --------------------
FINAL SINGLE POINT ENERGY         0.000000000000
-------------------------   --------------------



                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 0 seconds 0 msec
    """)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        exts = read_cis_output(sys.argv[1])
        if len(sys.argv) == 3:
            write_multiwfn_readable_orca_output_file(sys.argv[2], exts)
        else:
            write_multiwfn_readable_orca_output_file(sys.argv[1].replace(".out", "-orca.out"), exts)