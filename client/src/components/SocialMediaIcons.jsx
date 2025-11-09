import React from 'react'

const SocialMediaIcons = () => {
  return (
    <div className='flex justify-center md:justify-start my-10 gap-7'>
        <a href="https://linkdin.com" target='_blank' rel='noreffer' className='hover:opacity-50 transition duration-500'>
            <img src="../assets/linkedin.png" alt="linkdin" />
        </a>
        <a href="https://twitter.com" target='_blank' rel='noreffer' className='hover:opacity-50 transition duration-500'>
            <img src="../assets/twitter.png" alt="twitter" />
        </a>
        <a href="https://facebook.com" target='_blank' rel='noreffer' className='hover:opacity-50 transition duration-500'>
            <img src="../assets/facebook.png" alt="facebook" />
        </a>
        <a href="https://instagram.com" target='_blank' rel='noreffer' className='hover:opacity-50 transition duration-500'>
            <img src="../assets/instagram.png" alt="instagram" />
        </a>
    </div>
  )
}

export default SocialMediaIcons