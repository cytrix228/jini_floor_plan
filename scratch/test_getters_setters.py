import floorplan


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def show(label: str, value) -> None:
    print(f"{label}: {value}")


def main() -> None:
    initial_vtxl2xy = [0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]
    initial_site2xy = [2.0, 2.0, 8.0, 8.0]
    initial_site2room = [0, 1]
    initial_site2xy2flag = [1.0, 0.0, 0.0, 1.0]
    initial_room2area_trg = [40.0, 60.0]
    initial_room_connections = [(0, 1)]

    ctx = floorplan.OptimizeContext(
        initial_vtxl2xy,
        initial_site2xy,
        initial_site2room,
        initial_site2xy2flag,
        initial_room2area_trg,
        initial_room_connections,
        None,
    )

    section("Initial getter outputs")
    show("vtxl2xy", ctx.vtxl2xy())
    show("site2xy", ctx.site2xy())
    show("site2xy_ini", ctx.site2xy_ini())
    show("site2xy2flag", ctx.site2xy2flag())
    show("site2room", ctx.site2room())
    show("room2area_trg", ctx.room2area_trg())
    show("room_connections", ctx.room_connections())
    show("learning_rates", ctx.learning_rates())

    section("set_vtxl2xy / vtxl2xy")
    ctx.set_vtxl2xy([0.0, 0.0, 12.0, 0.0, 12.0, 12.0, 0.0, 12.0])
    show("vtxl2xy", ctx.vtxl2xy())

    section("set_site2xy / site2xy")
    ctx.set_site2xy([3.0, 2.5, 7.5, 8.5])
    show("site2xy", ctx.site2xy())

    section("set_site2xy_ini / site2xy_ini")
    ctx.set_site2xy_ini([1.0, 1.0, 9.0, 9.0])
    show("site2xy_ini", ctx.site2xy_ini())

    section("set_site2xy2flag / site2xy2flag")
    ctx.set_site2xy2flag([1.0, 1.0, 1.0, 1.0])
    show("site2xy2flag", ctx.site2xy2flag())

    section("set_site2room / site2room")
    ctx.set_site2room([1, 0])
    show("site2room", ctx.site2room())

    section("set_room2area_trg / room2area_trg")
    ctx.set_room2area_trg([45.0, 55.0])
    show("room2area_trg", ctx.room2area_trg())

    section("set_room_connections / room_connections")
    ctx.set_room_connections([(1, 0)])
    show("room_connections", ctx.room_connections())

    section("set_learning_rate / learning_rates")
    show("learning_rates (before)", ctx.learning_rates())
    ctx.set_learning_rate(0.005)
    show(
        "learning_rates (after)",
        ctx.learning_rates(),
    )
    print(
        "Note: set_learning_rate updates the optimizer state but does not mutate "
        "the params tuple returned by learning_rates().",
    )


if __name__ == "__main__":
    main()
