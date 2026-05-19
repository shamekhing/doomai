ACTION_NAMES = {
    0: "noop",
    1: "move_forward",
    2: "attack",
    3: "move_left",
    4: "move_right",
    5: "turn_left",
    6: "turn_right",
    7: "jump",
}


def pygame_action(pygame) -> int:
    keys = pygame.key.get_pressed()
    mouse_buttons = pygame.mouse.get_pressed(num_buttons=3)
    if keys[pygame.K_SPACE] or mouse_buttons[0]:
        return 2
    if keys[pygame.K_LEFT]:
        return 5
    if keys[pygame.K_RIGHT]:
        return 6
    if keys[pygame.K_a]:
        return 3
    if keys[pygame.K_d]:
        return 4
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return 1
    if keys[pygame.K_j]:
        return 7
    return 0
