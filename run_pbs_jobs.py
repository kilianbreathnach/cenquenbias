import subprocess

samples = ["18", "19", "20_a", "20_b"]
varnames = ["Re", "surf", "vdisp"]

# loop over each of the mass samples
#for i in range(1, 5):
for i in [1, 2, 4]:

    # and then over each of the variable configurations
    #for j in range(1, 8):
    for j in [7]: # [1, 2, 3, 4, 5, 6]:

        # generate the process string for the sample and variables
        binswitch = [int(b) for b in format(j, '03b')]
        varstring = [varnames[k] for k in range(3) if binswitch[k]]
        procstring = samples[i - 1] + "-" + "_".join(varstring)

        # open the pbs script file to edit
        fold = open("run_mcmc.pbs", 'r')
        fnew = open("run_mcmc.pbs.new", 'w')

        # iterate through the lines
        for line in fold:

            # rename the process
            if "#PBS -N" in line[:8]:
                newline = line.split()
                newline = " ".join(newline[:-1] + [procstring])
                fnew.write(newline + "\n")
            # change output and error filenames
            elif "#PBS -o" in line[:8]:
                newline = line.split("/")
                newline = "/".join(newline[:-1] + [procstring + ".out"])
                fnew.write(newline + "\n")
            elif "#PBS -e" in line[:8]:
                newline = line.split("/")
                newline = "/".join(newline[:-1] + [procstring + ".err"])
                fnew.write(newline + "\n")
            # change arguments to julia script
            elif "julia" in line[:7]:
                newline = line.split()
                newline = " ".join(newline[:2] +
                                   [str(i), str(j)] +
                                   newline[-2:])
                fnew.write(newline + "\n")
            # otherwise copy old line
            else:
                fnew.write(line)

        fold.close()
        fnew.close()

        # replace old file with new file
        subprocess.call(["mv", "run_mcmc.pbs.new", "run_mcmc.pbs"])

        # and now to submit the job to PBS
        subprocess.call(["qsub", "-k", "oe", "run_mcmc.pbs"])
