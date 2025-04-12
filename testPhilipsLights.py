from phue import Bridge
import time

# Set the IP address of your Philips Hue Bridge. Change this to match your device.
BRIDGE_IP = "192.168.178.197"  # Replace with your bridge IP

def main():
    # Connect to the Hue Bridge
    print("Connecting to Hue Bridge at", BRIDGE_IP)
    b = Bridge(BRIDGE_IP)

    # If the bridge is not registered, this command will raise an error.
    # Uncomment the next line if you need to force registration (then press the physical button on your bridge).
    # b.connect()

    # Get and print the available lights.
    lights = b.get_light_objects('id')
    print("Available Lights:")
    for light_id, light in lights.items():
        print(f"ID: {light_id} - Name: {light.name}")

    # Choose two lights to control.
    # Change these IDs to match your lights (you can see available IDs in the output above).
    light_ids = [1, 2]

    # Turn on the selected lights and set brightness to maximum.
    print("Turning on selected lights...")
    for light_id in light_ids:
        b.set_light(light_id, 'on', True)
        b.set_light(light_id, 'bri', 254)

    # Keep the lights on for 5 seconds.
    time.sleep(5)

    # Turn off the selected lights.
    print("Turning off selected lights...")
    for light_id in light_ids:
        b.set_light(light_id, 'on', False)

if __name__ == '__main__':
    main()
