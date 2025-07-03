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

                output = '{0:.2f} C, {1:.2f} hPa, {2:.2f} %RH'.format(
                    sensor.data.temperature,
                    sensor.data.pressure,
                    sensor.data.humidity
                )
                if sensor.data.heat_stable:
                    output += f", {row['gas']:.2f} Ohms"
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
