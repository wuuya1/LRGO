import os
import json
import glob
import imageio
import numpy as np
import pandas as pd

from plt2d import plot_episode

case_name = 'orca'  # circle_case rand_case
# safety_weight = 0
# forname = 2 * safety_weight

abs_path = os.path.abspath('.')
plot_save_dir = abs_path + '/' + case_name + '/' + 'animations' + '/'
log_save_dir = abs_path + '/' + case_name + '/'


def get_agent_traj(info_file, traj_file):
    with open(info_file, "r") as f:
        json_info = json.load(f)

    traj_list = []
    step_num_list = []
    agents_info = json_info['all_agent_info']
    targets_info = json_info['all_target']
    obstacles_info = json_info['all_obstacle']
    task_scheme = json_info['assignment_results']
    objs_info = {'targets_info': targets_info, 'obstacles_info': obstacles_info}
    task_area = json_info['TaskArea']
    exit_area = json_info['ExitArea']
    for agent_info in agents_info:
        agent_id = agent_info['id']
        agent_gp = agent_info['gp']
        agent_rd = agent_info['radius']

        df = pd.read_excel(traj_file, index_col=0, sheet_name='agent' + str(agent_id))
        step_num_list.append(df.shape[0])
        traj_list.append(df.to_dict('list'))
        # for indexs in df.index:
        #     traj_info.append(df.loc[indexs].values[:].tolist())
        # traj_info = np.array(traj_info)

    return agents_info, traj_list, step_num_list, objs_info, task_area, exit_area, task_scheme


def png_to_gif(general_fig_name, last_fig_name, animation_filename, video_filename):
    all_filenames = plot_save_dir + general_fig_name
    last_filename = plot_save_dir + last_fig_name
    print(all_filenames)

    # Dump all those images into a gif (sorted by timestep)
    filenames = glob.glob(all_filenames)
    filenames.sort()
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)
    for i in range(10):
        images.append(imageio.imread(last_filename))

    # Save the gif in a new animations sub-folder
    animation_save_dir = plot_save_dir + "animations/"
    os.makedirs(animation_save_dir, exist_ok=True)
    video_filename = animation_save_dir + video_filename
    animation_filename = animation_save_dir + animation_filename
    imageio.mimsave(animation_filename, images, )

    # convert .gif to .mp4
    try:
        import moviepy.editor as mp
    except imageio.core.fetching.NeedDownloadError:
        imageio.plugins.ffmpeg.download()
        import moviepy.editor as mp
    clip = mp.VideoFileClip(animation_filename)
    clip.write_videofile(video_filename)


info_file = log_save_dir + 'log/env_cfg.json'
traj_file = log_save_dir + 'log/trajs.xlsx'

agents_info, traj_list, step_num_list, tar_and_obs_info, task_area, exit_area, tasks_scheme = get_agent_traj(info_file,
                                                                                                             traj_file)

agent_num = len(step_num_list)
base_fig_name_style = "{test_case}_{policy}_{num_agents}agents"
base_fig_name = base_fig_name_style.format(policy=case_name, num_agents=agent_num, test_case=str(0).zfill(3))
general_fig_name = base_fig_name + '_*.png'
last_fig_name = base_fig_name + '.png'
animation_filename = base_fig_name + '.gif'
video_filename = base_fig_name + '.mp4'

plot_episode(tar_and_obs_info, agents_info, traj_list, step_num_list, plot_save_dir, base_fig_name,
             last_fig_name, task_area, exit_area, tasks_scheme, show=False)

png_to_gif(general_fig_name, last_fig_name, animation_filename, video_filename)
