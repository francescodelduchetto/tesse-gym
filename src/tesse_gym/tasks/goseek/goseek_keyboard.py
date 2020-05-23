###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from typing import Dict, List

from tesse.msgs import *
from tesse_gym.core.utils import NetworkConfig
from tesse_gym.eval.benchmark import Benchmark
from tesse_gym.tasks.goseek.goseek_full_perception import GoSeekFullPerception

import os
import numpy as np
import curses 
import cv2
# import time
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue


EVALUATION_METRICS = ["found_targets", "precision", "recall", "collisions", "steps"]


class GoSeekKeyboard(Benchmark):
    def __init__(
        self,
        scenes: List[int],
        episode_length: List[int],
        n_targets: List[int],
        build_path: str,
        network_config: NetworkConfig,
        success_dist: int,
        random_seeds: List[bool] = None,
        ground_truth_mode: bool = True,
    ):
        """ Configure evaluation.

        Args:
            scenes (List[int]): Scene IDs.
            episode_length (List[int]): Episode length.
            n_targets (List[int]): Number of targets.
            build_path (str): Path to TESSE build.
            success_dist (int): Maximum distance from target to be considered found.
            random_seeds (List[int]): Optional random seeds for each episode.
        """
        super().__init__()
        self.scenes = np.random.randint(1, 6, 50)# scenes
        self.episode_length = [400]*len(self.scenes)#episode_length
        self.random_seeds = random_seeds
        self.n_targets = [30]*len(self.scenes)#n_targets
        self.env = GoSeekFullPerception(
            build_path=build_path,
            network_config=network_config,
            scene_id=self.scenes[0],
            success_dist=success_dist,
            n_targets=n_targets[0],
            episode_length=max(episode_length),
            step_rate=self.STEP_RATE,
            ground_truth_mode=ground_truth_mode,
        )
        self.SHOW_MATPLOT = False
        self.SAVE_OBSERVATIONS = True

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """ Evaluate keyboard agent.

            Returns:
                Dict[str, Dict[str, float]]: Results for each scene
                    and an overall performance summary.
            """
        def plot_thread(image_queue):
            fig, ax = plt.subplots(1, 1)
            plt.ion()
            plt.show()
            print("plot process started")
            while True:
                print("asdsds")
                seg_img, done = image_queue.get()
                if done:
                    print("plot process end")
                    break
                ax.imshow(seg_img, vmin=0., vmax=1.)
                # fig.canvas.draw()
                plt.draw()
                plt.pause(0.0001)

        try:
            results = {}
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)
            
            if self.SHOW_MATPLOT:
                img_q = Queue()
                thread = Process(target=plot_thread, args=(img_q,))
                thread.start()
            for episode in range(len(self.scenes)):
                print(
                    f"Evaluation episode on episode {episode}, scene {self.scenes[episode]}"
                )
                n_found_targets = 0
                n_predictions = 0
                n_successful_predictions = 0
                n_collisions = 0
                step = 0

                self.env.n_targets = self.n_targets[episode]

                obs = self.env.reset(
                    scene_id=self.scenes[episode], random_seed=np.random.randint(0, 1000000)#self.random_seeds[episode]#
                )
                if self.SHOW_MATPLOT:
                    # plt.ion()
                    img_shape = (240, 320, 5)
                    imgs = np.reshape(obs[:-3], img_shape)
                    seg_img = cv2.resize(imgs[:, :, 3], (40, 40), interpolation=cv2.INTER_NEAREST)
                    img_q.put((seg_img, False))

                
                room = "none" # 0: no-fruit room, 1: fruit room, 2: corridor
                # for step in tqdm.tqdm(range(self.episode_length[episode])):
                for step in range(self.episode_length[episode]):
                    # pose = observation[:, -3:]
                    # plt.cla()
                    # plt.show()

                    skip = False

                    ch = stdscr.getkey()
                    # print(ch)
                    if ch == "KEY_UP":
                        action = 0
                    elif ch == "KEY_RIGHT":
                        action = 1
                    elif ch == "KEY_LEFT":
                        action = 2
                    elif ch == " ":
                        action = 3
                    elif ch == "0":
                        room = "non-fruit"
                        skip = True
                    elif ch == "1":
                        room = "fruit"
                        skip = True
                    elif ch == "2":
                        room = "corridor"
                        skip = True
                    else:
                        skip = True


                    if self.SAVE_OBSERVATIONS:
                        np.save(os.path.join("/tmp/observations", "obs_%s_%s_%s.npy"%(room, episode, step)), obs)

                    if skip:
                        continue
                    
                    obs, reward, done, info = self.env.step(action)


                    if self.SHOW_MATPLOT:
                        imgs = np.reshape(obs[:-3], img_shape)
                        seg_img = cv2.resize(imgs[:, :, 3], (40, 40), interpolation=cv2.INTER_NEAREST)
                        img_q.put((seg_img, False))

                    n_found_targets += info["n_found_targets"]

                    if action == 3:
                        n_predictions += 1
                        n_successful_predictions += 1 if info["n_found_targets"] else 0
                    if info["collision"]:
                        n_collisions += 1
                    if done:
                        break

                precision = (
                    1 if n_predictions == 0 else n_successful_predictions / n_predictions
                )
                recall = n_found_targets / self.env.n_targets
                results[str(episode)] = {
                    "found_targets": n_found_targets,
                    "precision": precision,
                    "recall": recall,
                    "collisions": n_collisions,
                    "steps": step + 1,
                }

            self.env.close()


            # combine scene results
            results["total"] = {
                metric: sum([scene_result[metric] for scene_result in results.values()])
                for metric in EVALUATION_METRICS
            }

            for metric in ["precision", "recall"]:
                results["total"][metric] /= len(self.scenes)

        finally:
            if self.SHOW_MATPLOT:
                img_q.put((None, True))
                thread.join()

            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.endwin()


        return results
