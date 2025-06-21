# cameras/management/commands/manage_test_videos.py

import os
import shutil
import glob
from django.core.management.base import BaseCommand
from django.conf import settings
from utils.enhanced_video_processor import EnhancedVideoProcessor

class Command(BaseCommand):
    help = 'Manage test videos for detection system'
    
    def add_arguments(self, parser):
        parser.add_argument(
            'action',
            choices=['list', 'add', 'remove', 'clear', 'stats'],
            help='Action to perform: list, add, remove, clear, or stats'
        )
        parser.add_argument(
            '--file',
            type=str,
            help='Video file path (for add/remove actions)'
        )
        parser.add_argument(
            '--name',
            type=str,
            help='Name for the video file in test directory'
        )
        
    def handle(self, *args, **options):
        action = options['action']
        test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        
        # Ensure test video directory exists
        os.makedirs(test_video_dir, exist_ok=True)
        
        if action == 'list':
            self.list_videos(test_video_dir)
        elif action == 'add':
            self.add_video(test_video_dir, options['file'], options.get('name'))
        elif action == 'remove':
            self.remove_video(test_video_dir, options['file'])
        elif action == 'clear':
            self.clear_videos(test_video_dir)
        elif action == 'stats':
            self.show_stats()
    
    def list_videos(self, test_video_dir):
        """List all video files in test directory"""
        self.stdout.write(
            self.style.SUCCESS(f'Video files in {test_video_dir}:')
        )
        
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        video_files = []
        
        for extension in video_extensions:
            pattern = os.path.join(test_video_dir, extension)
            video_files.extend(glob.glob(pattern))
        
        if not video_files:
            self.stdout.write(
                self.style.WARNING('No video files found.')
            )
            self.stdout.write(
                self.style.WARNING('Camera detection will be used instead.')
            )
        else:
            for i, video_file in enumerate(video_files, 1):
                file_name = os.path.basename(video_file)
                file_size = os.path.getsize(video_file)
                file_size_mb = file_size / (1024 * 1024)
                
                self.stdout.write(
                    f'{i}. {file_name} ({file_size_mb:.2f} MB)'
                )
            
            self.stdout.write(
                self.style.SUCCESS(f'\nTotal: {len(video_files)} video file(s)')
            )
            self.stdout.write(
                self.style.WARNING('These videos will be processed with priority over camera streams.')
            )
    
    def add_video(self, test_video_dir, source_file, target_name):
        """Add a video file to test directory"""
        if not source_file:
            self.stdout.write(
                self.style.ERROR('Please specify a video file with --file')
            )
            return
        
        if not os.path.exists(source_file):
            self.stdout.write(
                self.style.ERROR(f'Source file not found: {source_file}')
            )
            return
        
        # Determine target filename
        if target_name:
            target_file = os.path.join(test_video_dir, target_name)
        else:
            target_file = os.path.join(test_video_dir, os.path.basename(source_file))
        
        try:
            # Copy file to test directory
            shutil.copy2(source_file, target_file)
            
            file_size = os.path.getsize(target_file)
            file_size_mb = file_size / (1024 * 1024)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully added video: {os.path.basename(target_file)} '
                    f'({file_size_mb:.2f} MB)'
                )
            )
            self.stdout.write(
                self.style.WARNING(
                    'The detection system will now prioritize this video file over camera streams.'
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error adding video file: {str(e)}')
            )
    
    def remove_video(self, test_video_dir, video_name):
        """Remove a video file from test directory"""
        if not video_name:
            self.stdout.write(
                self.style.ERROR('Please specify a video file name with --file')
            )
            return
        
        # Check if it's a full path or just filename
        if os.path.dirname(video_name):
            target_file = video_name
        else:
            target_file = os.path.join(test_video_dir, video_name)
        
        if not os.path.exists(target_file):
            self.stdout.write(
                self.style.ERROR(f'Video file not found: {video_name}')
            )
            return
        
        try:
            os.remove(target_file)
            self.stdout.write(
                self.style.SUCCESS(f'Successfully removed video: {video_name}')
            )
            
            # Check if any videos remain
            remaining_videos = []
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            for extension in video_extensions:
                pattern = os.path.join(test_video_dir, extension)
                remaining_videos.extend(glob.glob(pattern))
            
            if not remaining_videos:
                self.stdout.write(
                    self.style.WARNING(
                        'No video files remain. The system will switch to camera detection.'
                    )
                )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error removing video file: {str(e)}')
            )
    
    def clear_videos(self, test_video_dir):
        """Clear all video files from test directory"""
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
        video_files = []
        
        for extension in video_extensions:
            pattern = os.path.join(test_video_dir, extension)
            video_files.extend(glob.glob(pattern))
        
        if not video_files:
            self.stdout.write(
                self.style.WARNING('No video files to clear.')
            )
            return
        
        try:
            removed_count = 0
            for video_file in video_files:
                os.remove(video_file)
                removed_count += 1
            
            self.stdout.write(
                self.style.SUCCESS(f'Successfully removed {removed_count} video file(s)')
            )
            self.stdout.write(
                self.style.WARNING(
                    'Test directory cleared. The system will switch to camera detection.'
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error clearing video files: {str(e)}')
            )
    
    def show_stats(self):
        """Show statistics about test detections"""
        try:
            processor = EnhancedVideoProcessorExtension()
            stats = processor.get_test_detection_statistics()
            
            self.stdout.write(
                self.style.SUCCESS('Test Detection Statistics:')
            )
            self.stdout.write(f"Total test detections: {stats['total_test_detections']}")
            self.stdout.write(f"Total storage used: {stats['total_size_mb']:.2f} MB")
            
            if stats['by_type']:
                self.stdout.write('\nDetections by type:')
                for detection_type, count in stats['by_type'].items():
                    self.stdout.write(f"  {detection_type}: {count}")
            
            if stats['recent_detections']:
                self.stdout.write('\nRecent detections:')
                for detection in stats['recent_detections'][:5]:
                    self.stdout.write(
                        f"  {detection['detection_time']} - "
                        f"{detection['alert_type']} "
                        f"(confidence: {detection['confidence']:.2f}) "
                        f"from {detection['source_video']}"
                    )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error getting statistics: {str(e)}')
            )