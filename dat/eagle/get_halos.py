import eagleSQLtools as sql
import numpy as np
import pandas as pd


# This uses the eagleSqlTools module to connect to the database with your username and password.
# If the password is not given, the module will prompt for it.
con = sql.connect("kwalsh", password="YMGIj9N6")

query = "SELECT \
                halo.groupID, \
                halo.Group_M_Mean200 as Mh, \
                halo.Group_R_Mean200 as Rh, \
                halo.GroupCentreOfPotential_x, \
                halo.GroupCentreOfPotential_y, \
                halo.GroupCentreOfPotential_z, \
                halo.NumOfSubhalos, \
                gal.GalaxyId, \
                gal.LastProgID, \
                gal.TopLeafID, \
                gal.DescendantID, \
                gal.GroupID, \
                gal.GroupNumber, \
                gal.SubGroupNumber, \
                gal.SnapNum, \
                gal.CentreOfMass_x, \
                gal.CentreOfMass_y, \
                gal.CentreOfMass_z \
        FROM \
                RefL0100N1504_FOF as halo, \
        	RefL0100N1504_SubHalo as gal \
        WHERE \
        	gal.GroupID = halo.GroupID and \
                gal.Spurious = 0 and \
                gal.TopLeafID \
        IN (SELECT \
                    gal.TopLeafID \
            FROM \
                    RefL0100N1504_SubHalo as gal, \
                    RefL0100N1504_Aperture as ap \
            WHERE \
                    gal.SnapNum = 28 and \
                    gal.Spurious = 0 and \
        	    ap.Mass_Star > 1.0e9 and \
                    ap.ApertureSize = 30 and \
                    gal.GalaxyID = ap.GalaxyID)"

# Execute query.
dat = sql.execute_query(con, query)

df = pd.DataFrame.from_dict(dat)
df.to_csv("halos.dat")
