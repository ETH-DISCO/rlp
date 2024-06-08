#!/usr/bin/env python
from rlp import puzzle as rp
import pygame


def main():
    parser = rp.make_puzzle_parser()
    args = parser.parse_args()

    p = rp.Puzzle(
        puzzle=args.puzzle,
        width=args.size[0],
        height=args.size[1],
        arg=args.arg,
        headless=args.headless,
    )
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_q
            ):
                running = False
            else:
                p.process_event(event)
        p.proceed_animation()

    p.destroy()


if __name__ == "__main__":
    main()
