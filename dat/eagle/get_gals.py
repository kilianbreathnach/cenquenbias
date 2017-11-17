import eagleSQLtools as sql
import numpy as np
import pandas as pd


# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con = sql.connect("kwalsh", password="YMGIj9N6")

query = "SELECT \
                 gal.GalaxyId, \
                 gal.LastProgID, \
                 gal.TopLeafID, \
                 gal.DescendantID, \
                 gal.GroupID, \
                 gal.GroupNumber, \
                 gal.SubGroupNumber, \
                 gal.Image_box, \
                 gal.CentreOfMass_x, \
                 gal.CentreOfMass_y, \
                 gal.CentreOfMass_z, \
                 gal.CentreOfPotential_x, \
                 gal.CentreOfPotential_y, \
                 gal.CentreOfPotential_z, \
                 gal.GasSpin_x, \
                 gal.GasSpin_y, \
                 gal.GasSpin_z, \
                 gal.HalfMassRad_Star, \
                 gal.HalfMassProjRad_Star, \
                 gal.MassType_Star, \
                 gal.StarFormationRate, \
                 gal.StellarVelDisp, \
                 gal.TotalEnergy, \
                 gal.Velocity_x, \
                 gal.Velocity_y, \
                 gal.Velocity_z, \
                 gal.Vmax, \
                 gal.VmaxRadius, \
                 ap.VelDisp, \
                 ap.SFR, \
                 ap.Mass_Star, \
                 mag.u_nodust as M_u, \
                 mag.g_nodust as M_g, \
                 mag.r_nodust as M_r, \
                 mag.i_nodust as M_i, \
                 mag.z_nodust as M_z, \
                 size.R_halfmass30, \
                 size.R_halfmass30_projected, \
                 halo.Group_M_Mean200 as Mh, \
                 halo.Group_R_Mean200 as Rh, \
                 halo.GroupCentreOfPotential_x, \
                 halo.GroupCentreOfPotential_y, \
                 halo.GroupCentreOfPotential_z, \
                 halo.NumOfSubhalos \
         FROM \
         	RefL0100N1504_SubHalo as gal, \
         	RefL0100N1504_Aperture as ap, \
         	RefL0100N1504_Magnitudes as mag, \
                RefL0100N1504_Sizes as size, \
                RefL0100N1504_FOF as halo \
         WHERE \
         	gal.SnapNum = 28 and \
                gal.Spurious = 0 and \
         	ap.Mass_Star > 1.0e9 and \
         	ap.ApertureSize = 30 and \
         	gal.GalaxyID = mag.GalaxyID and \
         	gal.GalaxyID = ap.GalaxyID and \
         	gal.GalaxyID = size.GalaxyID and \
         	gal.GroupID = halo.GroupID"

# Execute query.
dat = sql.execute_query(con, query)

df = pd.DataFrame.from_dict(dat)
df.to_csv("nugals.dat")
