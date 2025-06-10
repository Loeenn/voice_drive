import carla
import random
import time
import json
import websocket

CONTROL_MAP = {
    "start": {"throttle": 0.0, "steer": 0.0, "brake": 0.0},
    "turn_left": {"throttle": 0.3, "steer": -1.0, "brake": 0.0},
    "turn_right": {"throttle": 0.3, "steer": 1.0, "brake": 0.0},
    "stop": {"throttle": 0.0, "steer": 0.0, "brake": 1.0},
}


def setup_carla_client(host="localhost", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    return client


def configure_world(client):
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.02
    world.apply_settings(settings)
    return world, settings


def setup_traffic_manager(client):
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    return tm


def spawn_vehicle(world):
    bp = random.choice(world.get_blueprint_library().filter('vehicle.*'))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.try_spawn_actor(bp, spawn_point)
    return vehicle


def spawn_bus_stops(world):
    bp = world.get_blueprint_library().find('static.prop.busstop')
    transforms = [carla.Transform(carla.Location(3233, 716, 15))]
    return [world.spawn_actor(bp, tr) for tr in transforms]


def connect_websocket(uri):
    ws = websocket.create_connection(uri, ping_interval=None)
    ws.settimeout(0.001)
    return ws


def get_command(ws):
    try:
        msg = ws.recv()
        return json.loads(msg).get("command")
    except websocket.WebSocketTimeoutException:
        return None


def apply_manual_control(vehicle, command):
    ctl = CONTROL_MAP[command]
    vehicle.apply_control(carla.VehicleControl(
        throttle=ctl["throttle"],
        steer=ctl["steer"],
        brake=ctl["brake"]
    ))


def update_spectator(spectator, vehicle):
    tf = vehicle.get_transform()
    offset = tf.get_forward_vector() * -6 + carla.Location(z=2)
    spectator.set_transform(carla.Transform(tf.location + offset, tf.rotation))


def cleanup(world, settings, ws, vehicle, stops):
    settings.synchronous_mode = False
    world.apply_settings(settings)
    ws.close()
    vehicle.destroy()
    for s in stops:
        s.destroy()


def main_loop(world, settings, tm, vehicle, ws, stops):
    spectator = world.get_spectator()
    last_command = None
    vehicle.set_autopilot(True, tm.get_port())

    while True:
        world.tick()
        cmd = get_command(ws)

        if cmd in CONTROL_MAP:
            last_command = cmd
            if cmd == "stop":
                vehicle.set_autopilot(False, tm.get_port())
            elif cmd == "start":
                vehicle.set_autopilot(True, tm.get_port())

        if last_command and last_command != "start":
            apply_manual_control(vehicle, last_command)

        update_spectator(spectator, vehicle)
        time.sleep(settings.fixed_delta_seconds)


def main():
    client = setup_carla_client()
    world, settings = configure_world(client)
    tm = setup_traffic_manager(client)

    vehicle = spawn_vehicle(world)
    if vehicle is None:
        print("Failed to spawn vehicle.")
        return

    stops = spawn_bus_stops(world)
    ws = connect_websocket("ws://localhost:4959/ws/audio")

    try:
        main_loop(world, settings, tm, vehicle, ws, stops)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(world, settings, ws, vehicle, stops)


if __name__ == "__main__":
    main()