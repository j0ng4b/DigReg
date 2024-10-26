import pygame as pg
import numpy as np

from knn import KNN


SCREEN_TITLE = 'DigReg'

SCREEN_WIDTH = 480
SCREEN_HEIGHT = 320

SCREEN_FRAME_RATE = 60

LEFT_MOUSE_BUTTON = 1
MIDDLE_MOUSE_BUTTON = 2
RIGHT_MOUSE_BUTTON = 3

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_mnist(images_path, labels_path):
    with open(images_path, 'rb') as f:
        # Skip the magic number, number of images, number of rows and columns
        f.read(16)
        images = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28) / 255.0
    
    with open(labels_path, 'rb') as f:
        # Skip the magic number and number of labels
        f.read(8)
        labels = np.frombuffer(f.read(), np.uint8)
    
    return images.reshape(images.shape[0], -1), labels

def preprocess_canvas(canvas):
    pixel_array = pg.surfarray.array3d(canvas)

    array = np.dot(pixel_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    #array = 254 - array
    #array = array.astype(np.float64)
    return array


def main():
    knn = KNN(10)
    knn.fit(*load_mnist('mnist/train-images-idx3-ubyte', 'mnist/train-labels-idx1-ubyte'))

    pg.init()
    pg.display.set_caption(SCREEN_TITLE)
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    drawing = None
    canvas = pg.surface.Surface((SCREEN_WIDTH / 2, SCREEN_HEIGHT))
    canvas.fill(WHITE)

    running = True
    while running:
        # Event handler section
        for event in pg.event.get():
            running = not event.type == pg.QUIT

            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == LEFT_MOUSE_BUTTON:
                    drawing = (event.pos[0] - (SCREEN_WIDTH / 2), event.pos[1])
            
            if event.type == pg.MOUSEMOTION:
                if drawing is not None:
                    current_pos = (event.pos[0] - (SCREEN_WIDTH / 2), event.pos[1])
                    pg.draw.line(canvas, BLACK, drawing, current_pos, 5)
                    drawing = current_pos
            
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == LEFT_MOUSE_BUTTON:
                    drawing = None
                
                if event.button == RIGHT_MOUSE_BUTTON:
                    canvas.fill(WHITE)
                    preprocess_canvas(canvas)


        # Draw section
        screen.fill(BLACK)
        screen.blit(canvas, (SCREEN_WIDTH / 2, 0))
        pg.display.flip()

    pg.quit()



if __name__ == '__main__':
    main()