import time
import subprocess
import psutil
import re

def measure_power_consumption():
    # Use 'powertop' to measure power consumption
    power_measurement_command = 'sudo powertop -t 10'  # Run for 10 seconds as an example
    powertop_output = subprocess.check_output(power_measurement_command, shell=True, text=True)

    # Extract relevant information from powertop output
    power_consumption_info = extract_power_consumption_info(powertop_output)

    # Display power consumption information
    print("Power Consumption Information:")
    print(f"Power Consumption: {power_consumption_info['Power consumption']}")
    print(f"Energy Consumed: {power_consumption_info['Energy consumed']}")

def extract_power_consumption_info(powertop_output):
    # Extract relevant information using regular expressions
    power_consumption_info = {}

    power_pattern = re.compile(r'Power consumption:.+?(\d+\.\d+)')
    energy_pattern = re.compile(r'Energy consumed:.+?(\d+\.\d+)')

    power_match = power_pattern.search(powertop_output)
    energy_match = energy_pattern.search(powertop_output)

    if power_match:
        power_consumption_info['Power consumption'] = f"{power_match.group(1)} W"
    else:
        power_consumption_info['Power consumption'] = "N/A"

    if energy_match:
        power_consumption_info['Energy consumed'] = f"{energy_match.group(1)} J"
    else:
        power_consumption_info['Energy consumed'] = "N/A"

    return power_consumption_info

def calculate_co2_emissions(power_consumption):
    # Replace 'co2_emission_factor' with the CO2 emission factor for your region and energy mix
    co2_emission_factor = 0.43  # Bordeaux
    return power_consumption * co2_emission_factor

def run_style_transfer(iteration):
    # Your style transfer code goes here
    python = "python3"
    gatys = f"./Gatys/main.py --iterations {iteration}"
    lifei = f"./Lifei/main.py --train-flag True --cuda-device-no 0 --imsize 256 --cropsize 240 --train-content ./imgs/content/ --train-style imgs/style/mondrian.jpg --save-path ./ --max-iter {iteration}"
    style_transfer_command = f"{python} {gatys} > log.txt 2> log_warn.txt"
    subprocess.run(style_transfer_command, shell=True)

def simulate_workload(iteration):
    # Simulate a realistic workload
    print(f"Simulating workload for iteration {iteration}...")
    time.sleep(5)

def main(iteration):
    total_execution_time = 0

    start_time = time.time()

    # Run style transfer code
    run_style_transfer(iteration)

    # Measure power consumption
    measure_power_consumption()

    # Calculate execution time
    execution_time = time.time() - start_time
    total_execution_time += execution_time

    # Monitor CPU usage
    cpu_usage = psutil.cpu_percent()

    # Simulate additional workload (adjust as needed)
    simulate_workload(iteration)

    # Output results
    print(f"Iteration {iteration} results:")
    print(f"   Execution Time: {execution_time} seconds")
    print(f"   CPU Usage: {cpu_usage}%")

    # Sleep to simulate a realistic workload (adjust as needed)
    time.sleep(2)

    average_execution_time = total_execution_time
    print(f"\nAverage Execution Time over {iteration} iterations: {average_execution_time} seconds")

    # Calculate total power consumption and CO2 emissions
    # (Note: You need to replace the placeholders with actual data or functions)
    total_power_consumption = 1000  # Placeholder value
    total_co2_emissions = calculate_co2_emissions(total_power_consumption)

    print(f"Total Power Consumption: {total_power_consumption} watts")
    print(f"Total CO2 Emissions: {total_co2_emissions} kgCO2")

if __name__ == "__main__":
    main(1000)
    
    # parser = argparse.ArgumentParser(description="Style Transfer Analysis Script")
    # parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run style transfer")
    # args = parser.parse_args()
    # NUM_ITER = args.iterations
