from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import numpy as np
import csv

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary
import randomTrips

# ------------------------------------------------------------
# Anthropometric sampling functions (Brazilian population statistics)
# ------------------------------------------------------------

def altura_brasileiro():
    return round(np.random.normal(1.66, 0.08), 2)

def largura_brasileiro():
    return round(np.random.normal(0.43, 0.05), 2)

def comprimento_brasileiro():
    return round(np.random.normal(0.24, 0.03), 2)


# ------------------------------------------------------------
# Simulation run
# ------------------------------------------------------------

def run():
    TOTAL_USERS = 50
    SIM_DURATION = 3649  # exact seconds
    dados_brasileiros = {
        i: {
            "altura": altura_brasileiro(),
            "largura": largura_brasileiro(),
            "comprimento": comprimento_brasileiro(),
        }
        for i in range(TOTAL_USERS)
    }

    # Output CSV file
    with open('dataset.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'x', 'y', 'o', 'height'])

        person_data = {}
        started_recording = False  # recording starts only when all 50 people are present

        while traci.simulation.getTime() < SIM_DURATION:
            traci.simulationStep()

            lstPeopleId = traci.person.getIDList()

            # Check if all 50 simultaneous users are present
            if not started_recording:
                if len(lstPeopleId) == TOTAL_USERS:
                    started_recording = True
                else:
                    continue  # not all persons have spawned yet

            # Record only after all 50 are in the simulation
            for personId in lstPeopleId:
                x, y = traci.person.getPosition(personId)
                orientation = traci.person.getAngle(personId)

                if personId not in person_data:
                    person_data[personId] = []

                person_data[personId].append([
                    x, y, orientation,
                    dados_brasileiros[int(personId)]['altura']
                ])

        # Save all data to CSV
        for personId, records in person_data.items():
            for record in records:
                writer.writerow([personId] + record)

    traci.close()


# ------------------------------------------------------------
# Options and SUMO initialization
# ------------------------------------------------------------

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    return optParser.parse_args()[0]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    options = get_options()

    if options.nogui:
        sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo"
    else:
        sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui"

    # ------------------------------------------------------------
    # Generate 50 simultaneous users (all with depart=0)
    # ------------------------------------------------------------
    randomTrips.main(randomTrips.get_options([
        '--net-file', 'C:/UFG/Dados/UFG 2025/data/osm.net.xml',
        '--route-file', 'osm.pedestrians.rou.xml',
        "-e", "50",                  # generate trips from 0 to 50 s
        "--pedestrians",
        "--intermediate", "1000",
        "--seed", "42",
    ]))

    # Start SUMO
    traci.start([
        sumoBinary,
        '-c', "C:/UFG/Dados/UFG 2025/data/osm.sumocfg",
        "--fcd-output", "fcd.xml",
        "--device.rerouting.probability", "1",
        "--device.rerouting.period", "3"
    ])

    run()
