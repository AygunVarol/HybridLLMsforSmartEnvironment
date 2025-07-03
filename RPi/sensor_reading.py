#!/usr/bin/env python
import time
import csv
import os
import requests
import threading
import queue
import bme680

LOCATION = "office"
CSV_FILE = f"{LOCATION}_measurements.csv"
SERVER_URL = "http://192.168.0.100:5000/sensor_data"

unsent_queue = queue.Queue()


def init_csv():
    """Ensure the CSV file exists and has a header."""
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "location",
                "temperature",
                "pressure",
                "humidity",
                "gas",
                "air_quality",
            ])


def append_to_csv(row):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row.get("timestamp"),
                row.get("location"),
                row.get("temperature"),
                row.get("pressure"),
                row.get("humidity"),
                row.get("gas"),
                row.get("air_quality"),
            ]
        )


def send_loop():
    """Background thread to send queued rows to the edge server."""
    while True:
        row = unsent_queue.get()
        try:
            response = requests.post(SERVER_URL, json=row, timeout=5)
            if response.status_code == 200:
                print("Server response:", response.text)
            else:
                print(f"Server error {response.status_code}")
                unsent_queue.put(row)
        except Exception as e:
            print("Error sending data to server:", e)
            unsent_queue.put(row)
            time.sleep(5)
        time.sleep(1)


def run_sensor_reading():
    print("Starting sensor reading thread, sending sensor data to:", SERVER_URL)
    print("Initializing sensor...")

    init_csv()
    sender_thread = threading.Thread(target=send_loop, daemon=True)
    sender_thread.start()

    # Attempt to initialize the sensor using the primary I2C address,
    # if that fails, try the secondary address.
    try:
        sensor = bme680.BME680(bme680.I2C_ADDR_PRIMARY)
    except (RuntimeError, IOError):
        sensor = bme680.BME680(bme680.I2C_ADDR_SECONDARY)

    # (Optional) Print calibration data.
    print('Calibration data:')
    for name in dir(sensor.calibration_data):
        if not name.startswith('_'):
            value = getattr(sensor.calibration_data, name)
            if isinstance(value, int):
                print(f'{name}: {value}')

    # Configure sensor oversampling and filtering.
    sensor.set_humidity_oversample(bme680.OS_2X)
    sensor.set_pressure_oversample(bme680.OS_4X)
    sensor.set_temperature_oversample(bme680.OS_8X)
    sensor.set_filter(bme680.FILTER_SIZE_3)
    sensor.set_gas_status(bme680.ENABLE_GAS_MEAS)

    # Show an initial reading (optional)
    print('\nInitial reading:')
    for name in dir(sensor.data):
        if not name.startswith('_'):
            value = getattr(sensor.data, name)
            print(f'{name}: {value}')

    # Configure the gas heater settings.
    sensor.set_gas_heater_temperature(320)
    sensor.set_gas_heater_duration(150)
    sensor.select_gas_heater_profile(0)

    # --- Indoor Air Quality Setup ---
    # Collect burn-in data to establish a gas baseline.
    start_time = time.time()
    burn_in_time = 300
    burn_in_data = []

    print('Collecting gas resistance burn-in data for 5 mins')
    while time.time() - start_time < burn_in_time:
        if sensor.get_sensor_data() and sensor.data.heat_stable:
            gas = sensor.data.gas_resistance
            burn_in_data.append(gas)
            print(f'Gas: {gas} Ohms')
            time.sleep(1)

    gas_baseline = sum(burn_in_data[-50:]) / 50.0
    hum_baseline = 40.0
    hum_weighting = 0.25
    print(
        'Gas baseline: {0} Ohms, humidity baseline: {1:.2f} %RH'.format(
            gas_baseline,
            hum_baseline,
        )
    )

    print('\nPolling sensor data and sending to server:')
    try:
        while True:
            start = time.time()
            if sensor.get_sensor_data():
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                row = {
                    'timestamp': timestamp,
                    'location': LOCATION,
                    'temperature': sensor.data.temperature,
                    'pressure': sensor.data.pressure,
                    'humidity': sensor.data.humidity,
                    'gas': sensor.data.gas_resistance if sensor.data.heat_stable else ''
                }

                if sensor.data.heat_stable:
                    gas = sensor.data.gas_resistance
                    hum = sensor.data.humidity
                    gas_offset = gas_baseline - gas
                    hum_offset = hum - hum_baseline

                    if hum_offset > 0:
                        hum_score = (100 - hum_baseline - hum_offset)
                        hum_score /= (100 - hum_baseline)
                        hum_score *= (hum_weighting * 100)
                    else:
                        hum_score = (hum_baseline + hum_offset)
                        hum_score /= hum_baseline
                        hum_score *= (hum_weighting * 100)

                    if gas_offset > 0:
                        gas_score = (gas / gas_baseline)
                        gas_score *= (100 - (hum_weighting * 100))
                    else:
                        gas_score = 100 - (hum_weighting * 100)

                    air_quality_score = hum_score + gas_score
                    row['air_quality'] = air_quality_score

                output = '{0:.2f} C, {1:.2f} hPa, {2:.2f} %RH'.format(
                    sensor.data.temperature,
                    sensor.data.pressure,
                    sensor.data.humidity
                )
                if sensor.data.heat_stable:
                    output += f", {row['gas']:.2f} Ohms"
                    if 'air_quality' in row:
                        output += f", AQ: {row['air_quality']:.2f}"
                print(output)

                append_to_csv(row)
                unsent_queue.put(row)

            elapsed = time.time() - start
            if elapsed < 1:
                time.sleep(1 - elapsed)
    except KeyboardInterrupt:
        print("Exiting sensor reading loop.")

if __name__ == '__main__':
    run_sensor_reading()
