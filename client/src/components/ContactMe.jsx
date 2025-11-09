import React from 'react'
import LineGradient from './LineGradient'
import { useForm } from 'react-hook-form'
import { motion } from 'framer-motion'
const ContactMe = () => {
  const {
    register, 
    trigger, 
    formState: {errors}
  } = useForm();
  const onSubmit = async(e) => {
    const isValid = await trigger(); 
    if(!isValid) {
        e.preventDefault();
    }
  }
  return (
    <section id="contact" className='-mt-20' >
        {/*Headings*/}
        <motion.div
        className='flex justify-center w-full'
            initial="hidden"
            whileInView="visible"
            viewport={{once: true, amount: 0.5}}
            transition={{duration: 0.5}}
            variants={{
                hidden: {opacity: 0, x:-50}, 
                visible: {opacity: 1, x:0}
            }}>
                <div>

                <p className='font-playfair font-semibold text-xl'><span className='text-yellow'>Contact Me</span> To get Started</p>
                <div className='flex md:justify-center my-5'>
                <LineGradient width='mx-auto w-full' />
                </div>
                </div>
            </motion.div>
            {/*Image and Form Section*/}
            <div className='md:flex md:justify-between gap-16 mt-5'>
                <motion.div
                className='basis-1/2 flex justify-center'
                initial="hidden"
                whileInView="visible"
                viewport={{once: true, amount: 0.5}}
                transition={{duration: 0.5}}
                variants={{
                    hidden: {opacity: 0, y:-50}, 
                    visible: {opacity: 1, y:0}
                }}>
                    <img src="../assets/contact-image.jpeg" alt="contactImage" />
                </motion.div>
                <motion.div
                className='basis-1/2 mt-10 md:mt-0'
                initial="hidden"
                whileInView="visible"
                viewport={{once: true, amount: 0.5}}
                transition={{delay:0.2,duration: 0.5}}
                variants={{
                    hidden: {opacity: 0, y:-50}, 
                    visible: {opacity: 1, y:0}
                }}>
                    <form 
                    target='_blank'
                    onSubmit={onSubmit}
                    action='https://formsubmit.co/d5780ef6e35e5d535ed725378830eda1'
                    method='POST'
                    >
                        <input type="text"
                        className='w-full bg-blue placeholder-opaque-black font-semibold p-3 text-black font-playfair'
                        placeholder='NAME'
                        {...register("name", {
                            required: true, 
                            maxLength: 16
                        })} />
                        {
                            errors.name && (
                                <p className='text-red mt-1'>
                                    {errors.name.type === 'required' && 'this field is required'}
                                    {errors.name.type === 'maxLength' && 'max length is 16 charachters'}
                                </p>
                            )
                        }
                        <input type="text"
                        className='w-full bg-blue placeholder-opaque-black font-semibold p-3 mt-5 text-black font-playfair'
                        placeholder='EMAIL'
                        {...register("email", {
                            required: true, 
                            pattern: /^[A-Z0-9._%-]+@[A-Z0-9.-]+[A-Z]{2,}$/i
                        })}  />
                        {
                            errors.email && (
                                <p className='text-red mt-1'>
                                    {errors.email.type === 'required' && 'this field is required'}
                                    {errors.email.type === 'pattern' && 'invalid Email Address'}
                                </p>
                            )
                        }
                        <textarea type="text"
                        rows="4"
                        cols="50"
                        className='w-full bg-blue placeholder-opaque-black font-semibold p-3 mt-5 text-black font-playfair'
                        placeholder='message'
                        {...register("message", {
                            required: true, 
                            maxLength:2000
                        })}  />
                        {
                            errors.message && (
                                <p className='text-red mt-1'>
                                    {errors.message.type === 'required' && 'this field is required'}
                                    {errors.message.type === 'maxLength' && 'maxLength is 2000 charchters'}
                                </p>
                            )
                        }
                        <button
                        type='submit'
                        className='p-5 bg-yellow font-semibold text-deep-blue mt-5 hover:bg-red
                        hover:text-white transition duration-500'>SEND ME A MESSAGE</button>
                    </form>
                </motion.div>
            </div>
    </section>
  )
}

export default ContactMe