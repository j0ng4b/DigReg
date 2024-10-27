import pygame as pg
import numpy as np

from knn import KNN


SCREEN_TITLE = 'DigReg'

SCREEN_WIDTH = 480
SCREEN_HALF_WIDTH = SCREEN_WIDTH // 2
SCREEN_HEIGHT = 320
SCREEN_HALF_HEIGHT = SCREEN_HEIGHT // 2
SCREEN_FRAME_RATE = 60

CANVAS_SIZE = 28
CANVAS_WIDTH_SCALE = CANVAS_SIZE / SCREEN_HALF_WIDTH
CANVAS_HEIGHT_SCALE = CANVAS_SIZE / SCREEN_HEIGHT

LEFT_MOUSE_BUTTON = 1
MIDDLE_MOUSE_BUTTON = 2
RIGHT_MOUSE_BUTTON = 3

PREDICTION_NO_STATE = 0
PREDICTION_STARTED = 1
PREDICTION_DONE = 2

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
    array = 254 - array
    array = array.astype(np.float64) / 255.0
    array = np.flip(array, axis=0)
    array = np.rot90(array, k=-1)
    array = array.flatten()

    return array


def main():
    knn = KNN(10)
    knn.fit(*load_mnist('mnist/train-images-idx3-ubyte', 'mnist/train-labels-idx1-ubyte'))

    pg.init()
    pg.display.set_caption(SCREEN_TITLE)
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    drawing = None
    canvas = pg.surface.Surface((CANVAS_SIZE, CANVAS_SIZE))
    scaled_canvas = pg.surface.Surface((SCREEN_HALF_WIDTH, SCREEN_HEIGHT))
    canvas.fill(WHITE)

    font = pg.font.Font(None, 42)
    text_predicting = font.render('Predicting...', True, WHITE)
    text_result = font.render('You draw', True, WHITE)

    prediction_state = PREDICTION_NO_STATE

    running = True
    while running:
        # Event handler section
        for event in pg.event.get():
            running = not event.type == pg.QUIT

            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == LEFT_MOUSE_BUTTON:
                    pos_x = (event.pos[0] - SCREEN_HALF_WIDTH) * CANVAS_WIDTH_SCALE
                    pos_y = event.pos[1] * CANVAS_HEIGHT_SCALE

                    drawing = (pos_x, pos_y)
                    prediction_state = PREDICTION_NO_STATE

            if event.type == pg.MOUSEMOTION:
                if drawing is not None:
                    pos_x = (event.pos[0] - SCREEN_HALF_WIDTH) * CANVAS_WIDTH_SCALE
                    pos_y = event.pos[1] * CANVAS_HEIGHT_SCALE

                    pg.draw.line(canvas, BLACK, drawing, (pos_x, pos_y), 2)
                    drawing = (pos_x, pos_y)
                    prediction_state = PREDICTION_NO_STATE

            if event.type == pg.MOUSEBUTTONUP:
                if event.button == LEFT_MOUSE_BUTTON:
                    drawing = None
                    prediction_state = PREDICTION_NO_STATE

                if event.button == RIGHT_MOUSE_BUTTON:
                    prediction_state = PREDICTION_STARTED

            if event.type == pg.KEYUP:
                if event.key == pg.K_RETURN:
                    prediction_state = PREDICTION_STARTED

        if prediction_state == PREDICTION_STARTED:
            knn.predict([preprocess_canvas(canvas)])[0]
            canvas.fill(WHITE)
        elif prediction_state == PREDICTION_DONE:
            pass

        # Draw section
        screen.fill(BLACK)

        pg.transform.scale(canvas, (SCREEN_HALF_WIDTH, SCREEN_HEIGHT), scaled_canvas)
        screen.blit(scaled_canvas, (SCREEN_HALF_WIDTH, 0))

        if prediction_state == PREDICTION_STARTED:
            text_rect = text_predicting.get_rect()
            text_rect.center = (SCREEN_HALF_WIDTH // 2, SCREEN_HALF_HEIGHT)
            screen.blit(text_predicting, text_rect)

        if prediction_state == PREDICTION_DONE:
            text_rect = text_result.get_rect()
            text_rect.center = (SCREEN_HALF_WIDTH // 2, 0)
            text_rect.y = 10
            screen.blit(text_result, text_rect)

        pg.display.flip()

    pg.quit()



if __name__ == '__main__':
    main()
